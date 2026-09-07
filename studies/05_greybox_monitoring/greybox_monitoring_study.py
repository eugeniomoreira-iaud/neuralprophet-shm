# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: neuralprophet_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Study 05 · Grey-box expectation and monitoring of the station 02 inclination
#
# A reading arrives every twenty minutes: inclination, air temperature,
# relative humidity, and, since February 2025, wall temperature and solar
# radiation. This study answers the question Study 04 posed and did not fully
# answer:
#
# > **Given the environment measured at this instant, is this inclination the
# > one the wall was expected to show — and if not, which kind of departure is
# > it?**
#
# This study inherits Study 01's compensated, once-anchored, spike-cleaned
# inclination record; Study 02's characterisation of the station and ERA5
# proxies; Study 03's measured delays, gains and directions of the couplings;
# and Study 04's own honestly-stated weak points, which are the reason this
# study exists. The manufacturer's thermal compensation is taken as given
# throughout — no raw-channel fit is made, and every gain learned here is a
# post-compensation wall response, labelled as such. Forecasting stays out of
# scope: Study 04 already answered it, finding that skill comes from the
# response's own memory and that the environment adds nothing distinguishable
# from zero beyond six hours.
#
# The analysis proceeds in the following movements, one phase of the design's
# §8 at a time, each opened only after its predecessor's checkpoint is shown
# to the user and approved:
#
# 1. **Data and regressor sets (Phase 1).** Loaders, clock check, upsampling
#    onto the study's grid, and the filling of short regressor gaps, across
#    the three regressor sets.
# 2. **Harmonic diagnostics (Phase 1b).** A spectral scan of the target and of
#    the post-regressor residual, the daily cycle's amplitude and phase
#    measured over eight years, and the rank of the daily-by-annual surface —
#    fixing the seasonal Fourier orders and the conditional-seasonality
#    weight curve before any model is fitted.
# 3. **Model A attribution on three sets (Phase 2).** Component shares,
#    learned gains beside Study 03's, trend and seasonal parameters, the
#    conditional-seasonality test, fold stability, and residual diagnostics.
# 4. **Current-era ladder (Phase 2b).** Wall temperature and the on-structure
#    pyranometer tested rung by rung on the current era, to say what each
#    buys over the on-structure set alone.
# 5. **Expectation and interval (Phase 3).** A rolling refit with rolling
#    conformal calibration, and interval metrics reported by days since
#    refit.
# 6. **Model B impulse response (Phase 4).** A learned lagged-regressor
#    response on air temperature and station radiation, confronted with
#    Study 03's imposed delay-and-time-constant operator.
# 7. **The monitor (Phase 5).** Reference statistics, prewhitening, the three
#    charts, alarm attribution, and detectability with injections re-sized to
#    the wall's measured daily response.
# 8. **Outage bridges (Phase 6).** The expected level at resumption from
#    proxies that kept recording, against the level observed, per outage and
#    per proxy set.
#
# Phase 0 (smoke tests and housekeeping) precedes these movements outside the
# notebook, and Phase 7 (the report) follows them; see
# docs/superpowers/specs/2026-09-05-study05-greybox-monitoring-design.md §8
# for the full phase and checkpoint sequence.

# %% [markdown]
# ## Imports and parameters
#
# All library operations live in `shmlib`. This notebook owns the data
# pointers and the choices that make this one study: the regressor sets, the
# harmonic-diagnostic settings, the two models, the monitor, and the outage
# bridges.

# %%
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from neuralprophet import set_random_seed

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

from shmlib import (coupling, figures, monitoring, prediction, proxies,
                    quality, site, solar, tables, viz)

warnings.filterwarnings('ignore')
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)
viz.apply_report_style()

# %% [markdown]
# ## Parameters · Paths and grid
#
# Data pointers and the study's native grid. Every path below is declared
# here and passed into `shmlib` loaders as an argument; none is hard-coded in
# a library function.
#
# ### Parameter Tuning Guidance
#
# **`ARCHIVE_CSV`** — path to Study 1's verdict-aware archive product, the
# on-structure sensor package at its native cadence; default
# `'../../data/interim/archive/gubbio_archive_20min.csv'`. Every study since
# Study 1 reads this file rather than a raw `.adc` file.
#
# **`STATION_CSV`** — path to the town ground-station export; default
# `'../../data/raw/proxies/meteosystem_gubbio.csv'`. Loaded through
# `shmlib.proxies.load_ground_station`.
#
# **`ERA5_CSV`** — path to the Oikolab ERA5 reanalysis export; default
# `'../../data/raw/proxies/oikolab_weather.csv'`. Loaded through
# `shmlib.proxies.load_era5`.
#
# **`OUTPUT_DIR`** — where every `GM_` table and figure is written; default
# `Path('outputs')`, gitignored and regenerated by running the notebook.
#
# **`NATIVE_FREQ`** — the study's working grid; default `'20min'`, the
# archive's native spacing and the rate at which a deployed system receives a
# reading. Model A and the monitor run on this grid; Model B may fall back to
# an hourly grid under `MODEL_B_FALLBACK_FREQ` if Phase 0 finds that
# NeuralProphet 0.8.0 cannot carry lagged regressors at `n_lags=0`.
#
# **`TARGET_COLUMN`** — the response; default `'inc_comp_cleaned'`, Study 1's
# compensated, once-anchored, spike-cleaned inclination product. No raw or
# intermediate channel is used (D1, §2.4).
#
# **`SPIKE_COLUMN`** — Study 1's interpolation flag; default `'inc_spike'`.
# Read alongside the target so an interpolated value can be told from a
# measured one wherever that distinction matters downstream.
#
# **`WINDOW_START`** — the first instant of the study window; default
# `'2018-07-26'`, the archive's first day (D1). Unlike Study 04, this study
# uses the whole record rather than one post-outage segment.
#
# **`WINDOW_END`** — the last instant of the study window; default `None`,
# meaning the archive's own end. Set explicitly only to freeze a result
# against a growing archive.

# %%
ARCHIVE_CSV = '../../data/interim/archive/gubbio_archive_20min.csv'
STATION_CSV = '../../data/raw/proxies/meteosystem_gubbio.csv'
ERA5_CSV = '../../data/raw/proxies/oikolab_weather.csv'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

NATIVE_FREQ = '20min'
TARGET_COLUMN = 'inc_comp_cleaned'
SPIKE_COLUMN = 'inc_spike'

WINDOW_START = '2018-07-26'
WINDOW_END = None            # None: the archive's own end

# %% [markdown]
# ## Parameters · Regressor sets
#
# One model specification runs through three regressor sets — on-structure,
# station and ERA5 — mapping each of three roles (air temperature, relative
# humidity, solar radiation) to its source column (D2, D3). Radiation is
# global horizontal in every set, delayed by `RADIATION_DELAY_H` and passed
# through `shmlib.coupling.thermal_operator`; air temperature and humidity
# enter instantaneous.
#
# ### Parameter Tuning Guidance
#
# **`STR_MAP_CURRENT`** — the current-era (post-2025-02-21) block-to-quantity
# map, copied verbatim from Study 04's own parameter cell
# (`04_neuralprophet_inclination_prediction/neuralprophet_inclination_prediction_study.py`
# lines 253-259); default maps `tair`, `sr`, `twall`, `rh` and `batt` to the
# archive's current-era `_ok`/`_filtered` columns. Passed to
# `shmlib.proxies.load_sensor_forcings` so this study's on-structure load is
# the same choice Study 04 already validated, rather than a second,
# independently drifting definition of "the current era's channels".
#
# **`STR_MAP_LEGACY`** — the legacy-era block-to-quantity map, copied
# verbatim from the same lines (260-263) of Study 04; default maps `tair`,
# `rh` and `batt` to the legacy block's station-prefixed `_ok` columns
# (`site.TARGET_STATION` resolves the block). No `sr` or `twall` entry: the
# legacy package carries neither channel, and `load_sensor_forcings` leaves
# an unmapped role out of its output rather than inventing one.
#
# **`REGRESSOR_SETS`** — dict of set name to a role-to-column mapping. Three
# keys, `'str'`, `'gs'`, `'era5'`. The `'str'` set reads `tair_str` and
# `rh_str`, `load_sensor_forcings`' own column names for the on-structure
# quantities joined across the two instrument eras by `join_eras`, and
# borrows the station's own radiation column `sr_gs` because the wall's own
# pyranometer does not exist before 2025-02-21 and Study 01 condemned 161 of
# its days after that (D2); every table that reports the `'str'` set states
# this borrowing. The `'gs'` and `'era5'` sets read `shmlib.proxies`' own
# suffixed columns (`{quantity}_gs`, `{quantity}_era5`) unchanged. Accepted
# values: any mapping of `{'tair', 'rh', 'sr'}` to a column present in the
# joined frame; default as given in D2's table. Changing a mapping changes
# which columns Phase 1's loaders are asked to join, and therefore every
# downstream gain and coverage number for that set.
#
# **`RADIATION_DELAY_H`** — the delay applied to every radiation source
# before it enters Model A, in hours; default `1`, the diurnal-band optimum
# measured in Study 03 for all three radiation sources (D3). Applied through
# `shmlib.coupling.thermal_operator`, never inside the model itself.
#
# **`REGRESSOR_FILL_MAX_GAP`** — the longest regressor gap filled by linear
# interpolation before the row is dropped from a set's fit; default `'2h'`
# (D13). A regressor is a measured, smooth driver rather than the thing being
# judged, so short dust is filled and flagged; a longer gap leaves the row
# out of the fit for that set only. Raising it trades coverage for the risk
# of interpolating across a real excursion.
#
# **`ERA5_SR_IS_ACCUMULATION`** — whether the ERA5 radiation column is an
# accumulation over the preceding hour rather than an instantaneous reading;
# default `True` (D13). When `True`, the loader centres the value on the
# half-hour before interpolating onto the study's grid.

# %%
STR_MAP_CURRENT = {
    'tair': 'tair',
    'sr': 'n_sr_ok',
    'twall': 'n_twall_filtered',
    'rh': 'n_rh_ok',
    'batt': 'n_batt_ok',
}
STR_MAP_LEGACY = {
    'tair': 'tair',
    'rh': f'{site.TARGET_STATION}_rh_ok',
    'batt': f'{site.TARGET_STATION}_batt_ok',
}
REGRESSOR_SETS = {
    'str': {'tair': 'tair_str', 'rh': 'rh_str', 'sr': 'sr_gs'},   # sr borrowed from the station (D2)
    'gs': {'tair': 'tair_gs', 'rh': 'rh_gs', 'sr': 'sr_gs'},
    'era5': {'tair': 'tair_era5', 'rh': 'rh_era5', 'sr': 'sr_era5'},
}
RADIATION_DELAY_H = 1
REGRESSOR_FILL_MAX_GAP = '2h'
ERA5_SR_IS_ACCUMULATION = True

# %% [markdown]
# ## Parameters · Harmonic diagnostics (Phase 1b)
#
# Before any model is fitted, the seasonal Fourier orders and the
# conditional daily term's weight curve are fixed by measurement rather than
# by guess (D6). Two series are scanned: the target itself, and the residual
# of a plain least-squares regression of the target on the on-structure
# regressors — what the seasonal terms will actually have to explain.
#
# ### Parameter Tuning Guidance
#
# **`PERIOD_SCAN_BANDS`** — the period ranges the Lomb–Scargle scan
# certifies, each scanned and ranked on its own; default
# `{'short': (0.4, 3.0), 'long': (30.0, 900.0)}`. A single ranking spanning
# both bands lets the long-period continuum bury the short one: the
# sub-daily peaks carry a small share of the record's variance and are
# outranked by the annual-ish and multi-hundred-day peaks before a top-`k`
# cut ever gets to them, even when the sub-daily peaks are genuinely
# present. Scanning the bands separately gives each its own ranking. The
# `'short'` band brackets the daily and twelve-hour cycles; the `'long'`
# band brackets the annual and semi-annual ones. The grid edge `900.0`
# days is about a third of the eight-year record and is not itself a
# certified period — it exists so a slower structure is not aliased into
# the annual band.
#
# **`DAILY_HARMONIC_MIN_SLOTS`** — minimum number of the day's 72 slots
# required before a day's 24-hour harmonic is fitted; default `60`,
# matching Study 04's rule for a day informative enough to trust. A gappier
# day contributes no amplitude or phase estimate rather than a noisy one.
#
# **`ANNUAL_MODULATION_HARMONICS`** — Fourier orders tried when fitting the
# measured daily amplitude and phase against day of year; default `(1, 2)`.
# The order actually used is chosen on held-out years and reported in
# `GM_04`, never raised beyond what the scan supports.
#
# **`SURFACE_DOY_BINS`** — number of day-of-year bins the diurnal-band
# surface is binned into before its singular value decomposition; default
# `52`, one bin per week. Coarser binning trades resolution of the annual
# modulation for a less noisy surface.
#
# **`PERIOD_SCAN_N`** — number of log-spaced candidate periods scanned
# *per band* of `PERIOD_SCAN_BANDS`; default `40000`, which at one day
# gives a spacing of about `0.0002` days, inside the eight-year record's
# resolution, so the daily and twelve-hour peaks are sampled rather than
# stepped over within the short band. Fewer points make each band's scan
# faster and coarser.
#
# **`ANNUAL_MODULATION_MIN_GAIN`** — the fraction by which a higher Fourier
# order must lower the leave-one-year-out error before it is preferred over
# a lower one; default `0.01`. A value of zero returns to the bare
# lowest-error choice, which on a nearly sinusoidal modulation is decided
# by noise.

# %%
PERIOD_SCAN_BANDS = {'short': (0.4, 3.0), 'long': (30.0, 900.0)}
DAILY_HARMONIC_MIN_SLOTS = 60
ANNUAL_MODULATION_HARMONICS = (1, 2)
SURFACE_DOY_BINS = 52
PERIOD_SCAN_N = 40000
ANNUAL_MODULATION_MIN_GAIN = 0.01

# %% [markdown]
# ## Parameters · Model A
#
# The additive grey-box specification without autoregression: a trend, an
# annual term, a conditionally-weighted daily term, and the three regressor
# roles of one set, fitted with NeuralProphet's own quantile regression
# (D7).
#
# ### Parameter Tuning Guidance
#
# **`N_CHANGEPOINTS`** — trend changepoints, placed on covered time through
# `prediction.covered_changepoints`; default `12`, Study 04's value, swept
# in Phase 2 against the whole-record window rather than assumed unchanged.
#
# **`CHANGEPOINTS_RANGE`** — fraction of the training range eligible to
# carry a changepoint; default `0.95`.
#
# **`TREND_REG`** — regularization on the trend's rate changes; default
# `0.0`, the value `GM_05c`'s sweep chose on the on-structure set's held-out
# tail (D7, NeuralProphet tutorial 02 and the sub-daily guide). `None`
# defers to the sweep instead of fixing a value; the sweep cell still runs
# and reports every candidate's held-out MAE regardless.
#
# **`TREND_REG_CANDIDATES`** — the trend-regularisation values swept on the
# on-structure set's held-out tail before Model A is fitted for attribution
# (`GM_05c`); default `(0.0, 0.5, 1.0, 2.0, 5.0)`. These candidates, like
# `TREND_REG` itself, are in the wrapper's user-facing scale: NeuralProphet
# 0.8.0 rescales a positive value by `0.001` internally once changepoints
# exist, so the number stored on a fitted model's own
# `config_trend.trend_reg` is smaller than the candidate that produced it.
#
# **`YEARLY_ORDER`** — Fourier order of the annual term; default `1`, fixed
# from `GM_04` (D6): the regression residual certifies the annual cycle in its
# long band and no semi-annual one, so a second harmonic would have nothing to
# fit. Never raised beyond what the scan supports.
#
# **`DAILY_ORDER`** — Fourier order of the daily term(s); default `2`, fixed
# from `GM_04` (D6): both series certify the 12-hour companion of the daily
# cycle in the short band, the signature of a response that heats faster than
# it cools.
#
# **`WEEKLY_SEASONALITY`** — whether a weekly term is fitted; default
# `False`. Nature does not follow the week (D7).
#
# **`MODEL_A_LAGS`** — autoregressive lags; must stay `0`. Any positive
# value transfers the diurnal structure into the autoregressive term and
# makes the decomposition unreadable as physics (kept unchanged from Study
# 04, §2.4).
#
# **`QUANTILES`** — the interval's quantiles; default `(0.05, 0.95)`, a
# nominal 90 % interval. Must match `CONFORMAL_ALPHA` in the group below.
#
# **`LEARNING_RATE`, `EPOCHS`** — NeuralProphet's own training parameters;
# default `0.01` and `30`, Study 04's values.
#
# **`SEED`** — passed to `neuralprophet.set_random_seed` before every fit;
# default `0`, for a reproducible changepoint placement and initialization.
#
# **`MIN_TRAIN`** — shortest history a window must hold before it is
# allowed to speak; default `'730d'`, two full annual cycles — the minimum
# the model carrying an annual term can identify (D7).
#
# **`REFIT_EVERY`** — how often the rolling expectation refits; default
# `'30d'` (D5). Over thirty days the trend's extrapolation error stays a
# fraction of a millidegree at the measured drift rates, and the residual it
# leaves is stationary enough for the monitor to chart.
#
# **`TRAIN_WINDOW`** — the trailing history each refit is fitted on, rather
# than everything back to the record's start; default `'1095d'`, three
# years. Three annual cycles sit above `MIN_TRAIN`'s two, so the yearly term
# stays identifiable in every window, and a bounded window keeps every
# refit a comparable size across a multi-year record instead of the last
# fit dwarfing the first. This is also the "last N years" fallback the
# spec's risk table names for a walk-forward run that turns out too slow at
# an unbounded window — stated here in advance, rather than adopted only
# after a slow run is already under way.
#
# **`CONDITIONAL_DAILY`** — whether the daily term is fitted as two
# smoothly weighted conditional seasonalities (summer and winter shapes)
# rather than one plain daily term; default `True` (D7). Kept only if it
# improves held-out MAE or clears `CONDITIONAL_KEEP_MIN_SHARE`; the
# comparison is reported either way.
#
# **`WEIGHT_CURVE`** — the annual weight curve `w(doy)` blending the two
# conditional daily shapes; default the order-two annual Fourier fit of the
# regression residual's daily amplitude measured in `GM_04` (D6, D7): the
# coefficients are the constant, the first cosine and sine and the second
# cosine and sine over a period of 365.25 days, fitted on 1,752 days, and
# `prediction.seasonal_weights` normalises the curve to 0..1, where it peaks
# on day 206. The documented fallback, used only if the measured curve is no
# better on held-out folds, is the mid-July cosine
# `½(1 − cos(2π(doy − 15)/365))`, selected by setting this to `None`.
#
# **`CONDITIONAL_KEEP_MIN_SHARE`** — minimum variance share the conditional
# daily term must carry to be kept over the plain daily term when held-out
# MAE does not already decide it; default `0.01` (D7).
#
# **`STUDY03_GAINS`** — Study 03's measured gain per regressor set and
# driver, in millidegrees per unit of the driver, placed beside every
# learned gain in `GM_06`; default `{('str', 'tair'): -2.79, ('gs', 'tair'):
# -2.23, ('era5', 'tair'): -2.04, ('str', 'sr'): -0.035, ('gs', 'sr'):
# -0.035, ('era5', 'sr'): -0.026, ('str', 'rh'): np.nan, ('gs', 'rh'):
# np.nan, ('era5', 'rh'): np.nan}`. Relative humidity carries no Study 03
# measurement, hence `np.nan` on every set for that driver.
#
# **`SEASONAL_CURVE_DATES`** — calendar dates the daily term is drawn on in
# `GM_F05`/`GM_05d`; default `('2024-03-20', '2024-06-21', '2024-09-22',
# '2024-12-21')`, the equinoxes and solstices, the four points of the year
# at which a conditionally-weighted daily shape is most informatively
# compared. Only meaningful when the conditional term is kept
# (`CONDITIONS` is not `None`): when it is not, the plain daily term is the
# same shape on every day of the year, so drawing it four times would show
# four identical curves, and only the first date is evaluated instead.
#
# **`N_JOBS`** — worker processes `prediction.rolling_nowcast`,
# `fold_stability`, `sweep_trend_reg`, `attribution_fits` and
# `channel_ladder` fit their independent origins, folds, candidates, sets
# and rungs across; default `32`. `1` reproduces the serial run bit for
# bit (`_parallel_map` never imports `joblib` at that value). The useful
# ceiling is the length of the longest loop — about seventy walk-forward
# origins — rather than the machine's core count, and `_parallel_map`
# caps its pool at the number of items, so a value above the loop length
# only pays to start worker processes that go straight to idle. Each
# worker holds its own copy of the frame, about one gigabyte, so raising
# this alongside a much larger record is a memory decision as much as a
# speed one.

# %%
N_CHANGEPOINTS = 12
CHANGEPOINTS_RANGE = 0.95
TREND_REG = 0.0                # GM_05c: chosen by the sweep on the on-structure held-out tail
TREND_REG_CANDIDATES = (0.0, 0.5, 1.0, 2.0, 5.0)
YEARLY_ORDER = 1              # GM_04: annual peak certified on the residual, no semi-annual one
DAILY_ORDER = 2               # GM_04: the 12-hour companion of the daily cycle is certified
WEEKLY_SEASONALITY = False
MODEL_A_LAGS = 0
QUANTILES = (0.05, 0.95)
LEARNING_RATE = 0.01
EPOCHS = 30
SEED = 0
MIN_TRAIN = '730d'
REFIT_EVERY = '30d'
TRAIN_WINDOW = '1095d'
CONDITIONAL_DAILY = True
WEIGHT_CURVE = {               # GM_04: order-two annual fit of the residual's daily amplitude
    'order': 2,
    'coef': [2.620081844364288, -0.29217943337223173, -0.13479289899282565,
             0.2654898659335142, 0.26800490757467477],
    'period_days': 365.25,
    'n': 1752,
}                              # None selects the mid-July cosine fallback
CONDITIONAL_KEEP_MIN_SHARE = 0.01
STUDY03_GAINS = {('str', 'tair'): -2.79, ('gs', 'tair'): -2.23, ('era5', 'tair'): -2.04,
                 ('str', 'sr'): -0.035, ('gs', 'sr'): -0.035, ('era5', 'sr'): -0.026,
                 ('str', 'rh'): np.nan, ('gs', 'rh'): np.nan, ('era5', 'rh'): np.nan}
SEASONAL_CURVE_DATES = ('2024-03-20', '2024-06-21', '2024-09-22', '2024-12-21')
N_JOBS = 32

# %% [markdown]
# ## Parameters · Uncertainty and validation
#
# Quantile regression inside Model A, calibrated by rolling split conformal
# prediction, and NeuralProphet's own chronological folds for component
# stability (D8, §4.1).
#
# ### Parameter Tuning Guidance
#
# **`CONFORMAL_ALPHA`** — nominal miss rate of the conformal interval;
# default `0.10`, a nominal 90 % interval. Must match `QUANTILES` in the
# Model A group, or the coverage reported stops being comparable to the
# interval's own claim.
#
# **`CONFORMAL_METHOD`** — the conformal method passed to
# `m.conformal_predict`; default `'cqr'`, conformalized quantile regression
# (D8).
#
# **`CONFORMAL_CALIBRATION_WINDOW`** — how much of each refit's own
# out-of-sample residual history calibrates its interval; default
# `'180d'`, six months (D8). Study 04 calibrated once, on a stretch that
# ended eleven months before the record it was then asked to cover, and
# reached 67.7 % coverage; calibrating on each refit's own recent residuals
# is the fix this study makes.
#
# **`STALENESS_EDGES_D`** — bin edges, in days since a row's own refit
# origin, that group the rolling expectation's predictions by how stale
# their fit is when scored; default `[-0.01, 7, 14, 21, 31]`, four bins
# covering the first week, the second, the third and the run-out to
# `REFIT_EVERY`'s thirty days — the small negative first edge admits a
# same-day nowcast (`staleness_d == 0`) into the first bin rather than
# leaving it on a boundary. A study choice, not a library default: it says
# how finely Movement 3 wants to see the interval widen as a fit ages.
#
# **`VALID_P`** — fraction of each fold held out as NeuralProphet's own
# validation split; default `0.2`.
#
# **`CV_FOLDS`, `CV_FOLD_PCT`, `CV_FOLD_OVERLAP_PCT`** — the number,
# fractional size, and overlap of NeuralProphet's chronological
# cross-validation folds; default `5`, `0.1` and `0.0`. These are the folds
# component stability (gain, yearly amplitude, trend slope) is measured
# across (§4.1); a component that changes sign between them is not a
# finding, and Checkpoint 2 stops the study if the on-structure air-
# temperature gain is among them.

# %%
CONFORMAL_ALPHA = 0.10
CONFORMAL_METHOD = 'cqr'
CONFORMAL_CALIBRATION_WINDOW = '180d'
STALENESS_EDGES_D = [-0.01, 7, 14, 21, 31]
VALID_P = 0.2
CV_FOLDS = 5
CV_FOLD_PCT = 0.1
CV_FOLD_OVERLAP_PCT = 0.0

# %% [markdown]
# ## Parameters · Model B
#
# A learned lagged-regressor impulse response on the on-structure set, run
# alongside Model A to validate the delay-and-time-constant operator D3
# imposes rather than to forecast (D9).
#
# ### Parameter Tuning Guidance
#
# **`LAGGED_REGRESSORS`** — drivers entering through
# `add_lagged_regressor`; default `('tair', 'sr')`, the on-structure set's
# air temperature and its radiation role (`def_frames['str']`'s own `sr`
# column, the station's radiation borrowed under D2, since the wall's own
# pyranometer does not cover the study window), at the resolution Study
# 03's operator scan could not reach (D9). Humidity stays contemporaneous,
# outside this list.
#
# **`LAGGED_N_LAGS`** — history carried per lagged driver, in grid slots;
# default `36`, twelve hours at the native 20-minute grid (D9).
#
# **`LAGGED_REG`** — regularization on the lagged-regressor weights;
# default `None`, no sweep. Neither the spec (D9) nor the plan calls for
# one: this movement reads the learned weights as the impulse response
# itself, and a regulariser would smooth away the very shape the movement
# exists to measure.
#
# **`STUDY03_OPERATOR`** — Study 03's own delay-and-time-constant operator
# per driver, the reference the learned response is set beside in `GM_10`
# and `GM_F07`; default `{'tair': {'delay_h': 0.0, 'tau_h': 0.0}, 'sr':
# {'delay_h': 1.0, 'tau_h': 0.0}}` (spec §2.3): air temperature enters the
# wall's response instantaneously, radiation delayed one hour, and neither
# driver carries a fitted thermal time constant in Study 03's operator.
#
# **`MODEL_B_FALLBACK_FREQ`, `MODEL_B_FALLBACK_LAGS`** — the grid and lag
# count Model B falls back to if Phase 0 finds NeuralProphet 0.8.0 cannot
# carry lagged regressors at `n_lags=0`; default `'1h'` and `12`, one day
# of hourly history, and the sub-hour claim is dropped when this fallback is
# used (§9, risk row 1). Phase 0's
# `test_neuralprophet_capabilities.TestLaggedRegressorWithoutAutoregression`
# confirms lagged regressors do carry at `n_lags=0`, so this fallback is not
# taken.

# %%
LAGGED_REGRESSORS = ('tair', 'sr')
LAGGED_N_LAGS = 36
LAGGED_REG = None              # no sweep (D9): the weights are read as the response itself
STUDY03_OPERATOR = {'tair': {'delay_h': 0.0, 'tau_h': 0.0},
                    'sr': {'delay_h': 1.0, 'tau_h': 0.0}}
MODEL_B_FALLBACK_FREQ = '1h'
MODEL_B_FALLBACK_LAGS = 12

# %% [markdown]
# ## Parameters · Current-era ladder
#
# Wall temperature and on-structure radiation exist only from the
# 2025-02-21 instrument change onward. Rather than carry them in the main
# line at the cost of coverage, they are tested rung by rung on the current
# era alone (D4).
#
# ### Parameter Tuning Guidance
#
# **`CURRENT_ERA_START`** — first instant of the ladder's window; default
# `'2025-02-21'`, the instrument change (D4).
#
# **`TWALL_TAU_H`** — the wall-temperature time constant applied at the
# deployable rung of the ladder; default `4`, Study 03's measured value on
# the level band (§2.3).
#
# **`TWALL_LEAD_H`** — the wall temperature's measured lead over the
# deformation, in hours; default `-2` (negative: it leads). Used only on the
# diagnostic rung of the ladder, never for the expectation — a lead must be
# clamped to zero before any predictive use (§2.3, D4 rung 4).
#
# **`LADDER_BOOTSTRAP_BLOCK_HOURS`, `LADDER_BOOTSTRAP_REPETITIONS`** —
# block length and repetition count for the paired block-bootstrap MAE
# increment each rung reports; default `24` and `2000`, Study 04's values,
# since the residual this ladder scores is autocorrelated in the same way.
#
# **`LADDER_N_CHANGEPOINTS`** — trend changepoints allotted to each of the
# ladder's own fits; default `max(2, N_CHANGEPOINTS // 3)`. The ladder's
# window is a fifth of the main line's, so it is given a third of the main
# line's `N_CHANGEPOINTS`, and never fewer than two regardless of how small
# that leaves the count.
#
# **`LADDER_RUNGS`** — the four `(label, columns, gate)` triples
# `prediction.channel_ladder` fits in order: (1) the on-structure set
# alone, its radiation still borrowed from the ground station; (2) the same
# set with its own pyranometer in place of the borrowed radiation, so this
# rung differs from rung 1 in radiation source alone; (3) that set with the
# wall probe added through its measured thermal time constant
# (`TWALL_TAU_H`); and (4), a diagnostic only, the wall probe at its
# measured lead over the deformation (`TWALL_LEAD_H`) rather than its
# filtered value — a rung nobody can deploy, since it consumes the probe's
# own future readings, kept only to bound what the probe could buy under a
# lead that will never be available at prediction time. `gate` is the same
# three-column set — `['sr_wall', 'twall_tau', 'twall_lead']` — on every
# rung, including the first two, which do not themselves regress on the
# probe: D4 defines the ladder on **one** matched window, the rows where
# the pyranometer and the probe are both present, so that a rung's held-out
# error differs from the rung below's in its added channel alone and never
# in a training window that happens to be a different size. Gating rung 1
# on only `columns` (as an earlier version of this cell did) let it train
# on every current-era row while rungs 2-4 were held to the probe's and
# pyranometer's rows alone, so a chain skill computed across that boundary
# priced the channel and a halved training window together and could not
# be trusted. Default as given in D4's table.

# %%
CURRENT_ERA_START = '2025-02-21'
TWALL_TAU_H = 4
TWALL_LEAD_H = -2
LADDER_BOOTSTRAP_BLOCK_HOURS = 24
LADDER_BOOTSTRAP_REPETITIONS = 2000
LADDER_N_CHANGEPOINTS = max(2, N_CHANGEPOINTS // 3)
LADDER_MATCHED_GATE = ['sr_wall', 'twall_tau', 'twall_lead']
LADDER_RUNGS = [
    ('1 on-structure set', ['tair_str', 'rh_str', 'sr_gs'], LADDER_MATCHED_GATE),
    ('2 on-structure radiation', ['tair_str', 'rh_str', 'sr_wall'], LADDER_MATCHED_GATE),
    ('3 + twall, tau 4 h', ['tair_str', 'rh_str', 'sr_wall', 'twall_tau'],
     LADDER_MATCHED_GATE),
    ('4 twall at its lead (diagnostic)',
     ['tair_str', 'rh_str', 'sr_wall', 'twall_lead'], LADDER_MATCHED_GATE),
]

# %% [markdown]
# ## Parameters · Monitor
#
# Three control charts, one per damage mechanism and time scale, calibrated
# on a fixed reference window and swept for detectability with injections
# sized by the wall's measured daily response (D10, D11).
#
# ### Parameter Tuning Guidance
#
# **`REFERENCE_START`** — first instant of the in-control window the
# charts' statistics are calibrated on; default `'2020-11-21'`. The rolling
# residual Movement 3 produces exists only from its first refit origin
# onward — the on-structure set's first row plus `MIN_TRAIN`, printed by
# Movement 3 as 2020-11-20 10:40 — so a reference window starting in 2019,
# before any rolling residual is defined, would calibrate the charts on
# rows that do not exist. `REFERENCE_START` is instead the first full day
# the rolling residual covers.
#
# **`REFERENCE_END`, `MONITORED_START`** — the window's other bound and the
# first instant scored by the charts; default `'2021-12-31'` and
# `'2022-01-01'`, unchanged from the original design: the last full year
# before the 2022 outages and before the 2025 instrument change, with no
# outage longer than eleven days inside it (D10). Together with
# `REFERENCE_START` the reference window is 406 days, about thirteen
# months rather than the three full years first planned — the earliest the
# rolling residual allows — and the slow chart's one-year false-alarm
# budget therefore admits at most one false episode inside a window that
# short, so its tuning is necessarily coarse. Must stay in control:
# widening it to include a departure would calibrate the detector against
# the thing it is meant to find.
#
# **`EWMA_LAMBDA`, `EWMA_LAMBDA_DAILY`, `EWMA_LAMBDA_SLOW`** — EWMA
# smoothing constants for the fast, daily and slow charts; default `0.05`
# (Study 04's value), `0.2` and `0.1`. Smaller reacts more slowly and finds
# smaller sustained shifts; the daily and slow statistics already average
# over a day, so they carry less of their own noise and can afford a
# larger constant than the twenty-minute fast chart.
#
# **`CUSUM_K`, `CUSUM_H`** — CUSUM slack and decision interval, in standard
# deviations; default `0.5` and `5.0`, Study 04's values, shared by every
# chart.
#
# **`JOINT_WINDOW`, `JOINT_WINDOW_DAILY`** — coincidence window for the
# joint EWMA/CUSUM alarm; default `'6h'` for the fast chart and `'1D'` for
# the daily and slow charts, one native slot of their own daily grid.
#
# **`BUDGET_FAST_DAYS`, `BUDGET_DAILY_DAYS`, `BUDGET_SLOW_DAYS`** — the
# false-alarm budget each chart is tuned to, in watched days per false
# alarm on the in-control reference stretch; default `90.0`, `90.0` and
# `365.0` (D10's table). Fixed before any sweep; every detectability figure
# this study reports is only comparable at its chart's stated budget.
#
# **`LIMIT_CANDIDATES`** — control limits swept, in standard deviations, to
# find the smallest one meeting each chart's budget; default
# `tuple(np.arange(1.0, 15.01, 0.25))`. The floor was `2.0` in the first
# run, and `GM_12` showed the daily-phase chart tuned to exactly that
# floor — the sweep's own edge, not a genuine minimum, so the chart was
# less sensitive than its budget actually allowed. Lowering the floor to
# `1.0` gives the smallest-limit search room below the old edge; only a
# chart previously tuned to `2.00` can change, since the smallest limit
# meeting the budget is chosen and the achieved run length only rises with
# the limit. With the floor at `1.0` the daily-phase chart lands on the
# floor again, at the same run length of 203 days on the same two
# reference episodes: its joint alarm is gated by the CUSUM's decision
# interval (`CUSUM_H`), which every chart shares by D10, so below about
# two standard deviations the EWMA limit no longer sets the chart's
# sensitivity and lowering the floor further would change nothing. The
# floor is left at `1.0` and the report states the gate.
#
# **`DETECT_MAGNITUDES`, `DETECT_DURATIONS`** — injected amplitude-growth
# magnitudes (millidegrees) and durations swept for detectability; default
# `(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)` and `('6h', '24h', '72h', '168h',
# '336h')`, Study 04's grid, bracketing the residual's own scale. Shared by
# the amplitude, phase and step mechanisms; the drift mechanism uses
# `DETECT_DRIFT_HORIZONS` instead (below), since a drift needs weeks, not
# hours, to accumulate into anything a chart could see.
#
# **`DETECT_PHASE_SHIFTS_H`** — timing shifts probed for the phase
# mechanism, in hours; default `(0.25, 0.5, 1.0, 2.0)`. Each is sized by
# the wall's fitted daily response on the injection date rather than by the
# leftover daily seasonal term alone (D11) — the correction to Study 04,
# whose two-hour shift was scaled to 1.07 mdeg by the leftover term where
# the measured daily response implies about 5.6 mdeg (§2.5).
#
# **`DETECT_DRIFT_RATES`** — drift rates probed, in millidegrees per year;
# default `(1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)`. The first five
# are Study 04's own grid; the last three are added because the sweep at
# Study 04's rates and durations never crossed the slow chart's limit at
# all (`detected = 0` on every drift row of `GM_13`) — the injected drift
# never exceeded 0.8 mdeg inside the scoring window against a reference
# residual whose standard deviation is 13.4 mdeg. The added rates locate
# where the chart's threshold actually sits.
#
# **`DETECT_DRIFT_HORIZONS`** — the drift mechanism's own scoring
# durations, in place of `DETECT_DURATIONS`; default `('30d', '60d',
# '90d')`. `inject_anomaly(kind='drift')` takes its magnitude as a rate per
# year and accumulates continuously, so scoring it inside `DETECT_DURATIONS`'s
# longest span (336 hours, fourteen days) never lets a plausible drift rate
# accumulate into anything the slow chart's coarse limit would catch — the
# defect `DETECT_DRIFT_RATES` alone could not fix. All four injection dates
# plus ninety days plus `DETECT_RESPONSE_WINDOW` still fall inside the
# reference window (`2021-09-15` + 90 d = `2021-12-14`).
#
# **`DETECT_RESPONSE_WINDOW`** — how long after a departure ends an alarm
# still counts as having found it; default `'24h'`.
#
# **`DETECT_INJECTION_DATES`** — calendar dates injections are placed at;
# default `('2020-12-15', '2021-03-15', '2021-06-15', '2021-09-15')`, one
# per season, each chosen to fall inside the reference window fixed above
# with at least fourteen days of record ahead of it plus
# `DETECT_RESPONSE_WINDOW` before the window's own end, so a sweep never
# runs off the edge of the only stretch known to be in control (D11).
#
# **`MECHANISM_CHARTS`** — the chart and the statistic each damage
# mechanism is scored on; default `{'amplitude': ('daily_amplitude',
# 'daily_amplitude'), 'phase': ('daily_phase', 'daily_phase'), 'drift':
# ('slow', 'daily_mean'), 'step': ('fast', 'innovation')}` (D11): an
# amplitude growth and a phase shift are properties of the daily cycle and
# are read off the daily chart's two statistics, a drift is read off the
# slow chart's daily mean, and a sudden step is read off the fast chart's
# prewhitened innovations.
#
# **`ATTRIBUTION_CHANNELS`** — channels a fast alarm is cross-checked
# against before it is called a structural departure; default `('tair',
# 'rh', 'batt')` (D10). An alarm coincident with a swing on one of these is
# attributed to the environment or the instrument rather than to the wall.
#
# **`ATTRIBUTION_WINDOW`** — width of `channel_coincidence`'s centred
# rolling median, the baseline each channel's departure is measured
# against; default `'2h'`. The function's own default, `'24h'`, spans an
# entire diurnal cycle, so on air temperature the "departure from the
# median" is the diurnal cycle itself — a MAD of about 6 °C against a
# threshold near 31 °C that no realistic swing reaches, which is why every
# one of the first run's 75 fast-chart episodes came back `unattributed`.
# The coincidence test is built to catch a twenty-minute-scale swing
# against its own local background, not a slow cycle the median should
# already track out, so the window is shortened to two hours instead.
#
# **`ATTRIBUTION_THRESHOLD`** — number of scaled departures a channel must
# exceed to count as in excursion, passed to `channel_coincidence`;
# default `5.0`, the function's own default, unchanged — only the window
# needed correcting.

# %%
REFERENCE_START = '2020-11-21'
REFERENCE_END = '2021-12-31'
MONITORED_START = '2022-01-01'
EWMA_LAMBDA = 0.05
EWMA_LAMBDA_DAILY = 0.2
EWMA_LAMBDA_SLOW = 0.1
CUSUM_K = 0.5
CUSUM_H = 5.0
JOINT_WINDOW = '6h'
JOINT_WINDOW_DAILY = '1D'
BUDGET_FAST_DAYS = 90.0
BUDGET_DAILY_DAYS = 90.0
BUDGET_SLOW_DAYS = 365.0
LIMIT_CANDIDATES = tuple(np.arange(1.0, 15.01, 0.25))
DETECT_MAGNITUDES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DETECT_DURATIONS = ('6h', '24h', '72h', '168h', '336h')
DETECT_PHASE_SHIFTS_H = (0.25, 0.5, 1.0, 2.0)
DETECT_DRIFT_RATES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
DETECT_DRIFT_HORIZONS = ('30d', '60d', '90d')
DETECT_RESPONSE_WINDOW = '24h'
DETECT_INJECTION_DATES = ('2020-12-15', '2021-03-15', '2021-06-15', '2021-09-15')
MECHANISM_CHARTS = {'amplitude': ('daily_amplitude', 'daily_amplitude'),
                    'phase': ('daily_phase', 'daily_phase'),
                    'drift': ('slow', 'daily_mean'),
                    'step': ('fast', 'innovation')}
ATTRIBUTION_CHANNELS = ('tair', 'rh', 'batt')
ATTRIBUTION_WINDOW = '2h'
ATTRIBUTION_THRESHOLD = 5.0

# %% [markdown]
# ## Parameters · Outage bridges
#
# For each whole-day outage of seven days or more, a model refitted on data
# up to the outage predicts the level at resumption from the proxy sets that
# kept recording, and the observed mean over the days that follow is
# compared against it (D12). Nothing is written into the gap: an outage is a
# hypothesis to test, never data to fit on.
#
# ### Parameter Tuning Guidance
#
# **`OUTAGES`** — the seven whole-day outages of the archive, each as an
# `(start, end)` ISO date pair, both inclusive. Start dates and durations
# are `docs/data-quality-report-2026-08-10.md` §1's table; end dates are
# computed as `start + (duration - 1)` days, checked against the one end
# date already published — the 271-day outage's last missing day,
# 2023-06-20 (Study 04). Default the whole seven: 2020-06-06 to 2020-06-16
# (11 d); 2022-06-02 to 2022-09-21 (112 d); 2022-09-23 to 2023-06-20
# (271 d); 2024-05-05 to 2024-06-13 (40 d); 2024-08-28 to 2024-10-08
# (42 d); 2025-09-26 to 2025-10-12 (17 d); 2026-03-07 to 2026-06-17
# (103 d). Every outage in the archive is seven days or more, so all seven
# enter the bridge.
#
# **`OUTAGE_SETTLE_DAYS`** — days excluded at the start of the post-outage
# window before the observed mean is taken; default `1`, Study 01's restart
# transient.
#
# **`OUTAGE_WINDOW_DAYS`** — length of the post-outage window the observed
# mean is taken over; default `7` (D12).
#
# **`BRIDGE_SETS`** — which regressor sets predict across an outage;
# default `('gs', 'era5')`, the two sets whose proxies kept recording
# through every outage. The `'str'` set is excluded here by construction:
# the on-structure package is exactly what the outage took down.

# %%
OUTAGES = (
    ('2020-06-06', '2020-06-16'),   # 11 d
    ('2022-06-02', '2022-09-21'),   # 112 d
    ('2022-09-23', '2023-06-20'),   # 271 d
    ('2024-05-05', '2024-06-13'),   # 40 d
    ('2024-08-28', '2024-10-08'),   # 42 d
    ('2025-09-26', '2025-10-12'),   # 17 d
    ('2026-03-07', '2026-06-17'),   # 103 d
)
OUTAGE_SETTLE_DAYS = 1
OUTAGE_WINDOW_DAYS = 7
BRIDGE_SETS = ('gs', 'era5')

# %% [markdown]
# ## Movements
#
# No analysis cell exists yet. Cells are added phase by phase, in the order
# Phase 1, Phase 1b, Phase 2, Phase 2b, Phase 3, Phase 4, Phase 5, Phase 6
# above, each shown to the user as a checkpoint and approved before the next
# phase's cells are written. Phase 0 (smoke tests and housekeeping) runs
# before this notebook is touched, and Phase 7 (the report) is written
# afterwards, never in this file. See
# docs/superpowers/specs/2026-09-05-study05-greybox-monitoring-design.md §8
# for the full phase and checkpoint sequence.

# %% [markdown]
# ## Movement 1 · The record and the three regressor sets
#
# Study 1's product is read at its native cadence, the two external sources
# are brought onto the same grid, each source's clock is checked before
# anything is joined, and the three regressor sets of the design (D2) are
# assembled with Study 3's operator applied to radiation (D3). Nothing here
# fills the target; regressor dropouts up to `REGRESSOR_FILL_MAX_GAP` are
# filled and flagged (D13).

# %%
target, target_provenance = proxies.load_response(
    ARCHIVE_CSV, column=TARGET_COLUMN, spike_column=SPIKE_COLUMN,
    honour_spike=True, freq=NATIVE_FREQ, tz=site.SITE_TZ, min_count=1)
target = target.loc[WINDOW_START:WINDOW_END]

sensor_current, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_CURRENT, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor_legacy, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_LEGACY, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor = proxies.join_eras([sensor_current, sensor_legacy])

station_hourly = proxies.load_ground_station(STATION_CSV)
era5_hourly = proxies.load_era5(ERA5_CSV)
print(f'target {target.notna().sum():,} accepted slots of {len(target):,}; '
      f'station {len(station_hourly):,} h; ERA5 {len(era5_hourly):,} h')

# %% [markdown]
# ### The clock of each source
#
# Study 2's two-sided test: radiation against computed solar noon, and each
# source against ERA5 by cross-correlation. Run on the hourly grid, where
# the test was designed, before any upsampling.

# %%
hourly = proxies.harmonise(
    [proxies.load_sensor_forcings(ARCHIVE_CSV, column_map=STR_MAP_CURRENT,
                                  honour_suspect=True)[0],
     station_hourly, era5_hourly], freq=site.ANALYSIS_FREQ)
clock = quality.clock_check(hourly, 'sr')
display(clock)
clock.to_csv(OUTPUT_DIR / 'GM_03_clock_check.csv', index=False)
tables.write_table(clock, str(OUTPUT_DIR / 'GM_03_body.tex'),
                   [('source', tables.texttt), ('channel', tables.texttt),
                    ('n_days_solar', ',d'), ('solar_offset_h', '.2f'),
                    ('solar_offset_iqr_h', '.2f'), ('xcorr_lag_h', '.1f'),
                    ('xcorr_r', '.3f'), ('tests_agree', tables.yes_no)])

# %% [markdown]
# ### Onto the native grid, and the regressor sets

# %%
station = proxies.to_native_grid(station_hourly, freq=NATIVE_FREQ,
                                 accumulations=())
era5 = proxies.to_native_grid(
    era5_hourly, freq=NATIVE_FREQ,
    accumulations=('sr',) if ERA5_SR_IS_ACCUMULATION else ())
record = proxies.harmonise([sensor, station, era5, target.to_frame('y')],
                           freq=NATIVE_FREQ).loc[WINDOW_START:WINDOW_END]

sets, frame = proxies.build_regressor_sets(
    record, REGRESSOR_SETS, target='y', fill_max_gap=REGRESSOR_FILL_MAX_GAP,
    radiation_delay_h=RADIATION_DELAY_H, freq=NATIVE_FREQ)

# %% [markdown]
# ### Coverage and the anatomy of the target's gaps

# %%
coverage = proxies.regressor_set_coverage(sets, frame['y'], TARGET_COLUMN)
display(coverage)
coverage.to_csv(OUTPUT_DIR / 'GM_01_window_coverage.csv', index=False)
tables.write_table(coverage, str(OUTPUT_DIR / 'GM_01_body.tex'),
                   [('set', tables.texttt), ('role', tables.texttt),
                    ('accepted', ',d'), ('filled', ',d'),
                    ('coverage', tables.percent)])

gaps = prediction.gap_inventory(frame['y'], freq=NATIVE_FREQ)
gap_classes = (gaps.groupby('gap_class', sort=False)
               .agg(gaps=('n_slots', 'size'), hours=('duration_h', 'sum'))
               .reset_index())
display(gap_classes)
gaps.to_csv(OUTPUT_DIR / 'GM_02_gap_inventory.csv', index=False)
tables.write_table(gap_classes, str(OUTPUT_DIR / 'GM_02_body.tex'),
                   [('gap_class', tables.texttt), ('gaps', ',d'),
                    ('hours', ',.0f')])

# %%
figures.plot_regressor_sets(
    frame, 'y', {n: {r: f'{r}_{n}' for r in ['tair', 'rh', 'sr']}
                 for n in sets},
    target_channel='inc_comp',
    title='The record and the three regressor sets',
    save_path=str(OUTPUT_DIR), filename='GM_F01_regressor_sets')

# %% [markdown]
# ## Movement 1b · Harmonic diagnostics before modelling
#
# Two series: the target, and the residual of a plain least-squares
# regression of the target on the on-structure set's drivers. The spectral
# scan fixes the seasonal orders; the daily cycle's amplitude and phase,
# fitted against day of year, give the weight curve; the surface rank says
# how many weighted daily shapes the conditional term needs (D6).

# %%
residual_ols, ols_gains = prediction.ols_residual(
    frame['y'], sets['str'][['tair', 'rh', 'sr']])
series_for_scan = {'target': frame['y'], 'residual': residual_ols}
print('OLS gains on the on-structure set:',
      ols_gains.drop('intercept').round(4).to_dict())

# %%
scans, dailies, fits = {}, {}, {}
for name, series in series_for_scan.items():
    scans[name] = pd.concat(
        [prediction.period_scan(series, min_days=lo, max_days=hi,
                                n_periods=PERIOD_SCAN_N, spacing='log', top=5)
         .assign(series=name, band=band_name)
         for band_name, (lo, hi) in PERIOD_SCAN_BANDS.items()],
        ignore_index=True)
    band = coupling.diurnal_band(series, window=72)
    dailies[name] = monitoring.daily_harmonic(band, min_slots=DAILY_HARMONIC_MIN_SLOTS)
    dailies[name]['phase_h'] = coupling.centre_phase(dailies[name]['phase_h'])
    table_a, fit_a = coupling.annual_modulation(dailies[name]['amplitude'],
                                                harmonics=ANNUAL_MODULATION_HARMONICS,
                                                min_gain=ANNUAL_MODULATION_MIN_GAIN)
    table_p, fit_p = coupling.annual_modulation(dailies[name]['phase_h'],
                                                harmonics=ANNUAL_MODULATION_HARMONICS,
                                                min_gain=ANNUAL_MODULATION_MIN_GAIN)
    fits[name] = {'amplitude': fit_a, 'phase': fit_p,
                  'amplitude_table': table_a.assign(series=name, statistic='amplitude'),
                  'phase_table': table_p.assign(series=name, statistic='phase')}
surface = coupling.cycle_surface_rank(coupling.diurnal_band(residual_ols, window=72),
                                      doy_bins=SURFACE_DOY_BINS, freq=NATIVE_FREQ)

scan_table = pd.concat(scans.values(), ignore_index=True).assign(block='period_scan')
modulation_table = pd.concat(
    [fits[n][k] for n in fits for k in ('amplitude_table', 'phase_table')],
    ignore_index=True).assign(block='annual_modulation')
harmonic = pd.concat(
    [scan_table, modulation_table,
     surface['table'].assign(block='surface_rank', series='residual')],
    ignore_index=True)
display(harmonic)
harmonic.to_csv(OUTPUT_DIR / 'GM_04_harmonic_diagnostics.csv', index=False)
tables.write_table(
    scan_table, str(OUTPUT_DIR / 'GM_04_body.tex'),
    [('series', tables.texttt), ('band', tables.texttt), ('rank', 'd'),
     ('period_days', ',.4f'), ('power', '.4f')])
tables.write_table(
    modulation_table, str(OUTPUT_DIR / 'GM_04b_body.tex'),
    [('series', tables.texttt), ('statistic', tables.texttt), ('order', 'd'),
     ('holdout_mse', ',.3f'), ('chosen', tables.yes_no)])
tables.write_table(
    surface['table'].head(6), str(OUTPUT_DIR / 'GM_04c_body.tex'),
    [('component', '.0f'), ('variance_share', tables.percent),
     ('cumulative_share', tables.percent)])

figures.plot_harmonic_diagnostics(
    scans, dailies, fits, surface, title='Harmonic diagnostics',
    save_path=str(OUTPUT_DIR), filename='GM_F02_harmonic_diagnostics')

# %%
# What the diagnostic decides, printed so the checkpoint can read it back.
short = scans['residual'][scans['residual']['band'] == 'short']
long_ = scans['residual'][scans['residual']['band'] == 'long']
has_semi_annual = prediction.has_certified_period(long_, 182.6)
has_annual = prediction.has_certified_period(long_, 365.25)
has_twelve_hour = prediction.has_certified_period(short, 0.5, tolerance_days=0.05)
has_daily = prediction.has_certified_period(short, 1.0, tolerance_days=0.05)
print('annual peak on the residual:', bool(has_annual))
print('semi-annual peak on the residual:', bool(has_semi_annual))
print('12-hour peak on the residual:', bool(has_twelve_hour))
print('daily peak on the residual:', bool(has_daily))
print('annual modulation order for the daily amplitude:', fits['residual']['amplitude']['order'])
print('first two surface components carry',
      f"{surface['table']['cumulative_share'].iloc[1]:.1%}")
weights_measured = prediction.seasonal_weights(frame.index, modulation=fits['residual']['amplitude'])
print('measured summer weight peaks on day', int(weights_measured['summer_w'].idxmax().dayofyear))
for name in fits:
    print(name, 'amplitude fit:',
          {k: (v.tolist() if hasattr(v, 'tolist') else v)
           for k, v in fits[name]['amplitude'].items()})
    print(name, 'certified periods:')
    display(scans[name])

# %% [markdown]
# ## Movement 2 · What the record is made of
#
# One specification (D7), fitted once per regressor set on the full training
# window for attribution. The trend regularisation is swept on a held-out
# tail; the conditional daily term is tested against the plain one and kept
# only if it earns its place. Every native NeuralProphet plot runs here as a
# diagnostic, rendered to PNG rather than shown as inline SVG; the report's
# figures are the same content redrawn (D14). The movement writes the tables
# `GM_04d`, `GM_05`, `GM_05b`, `GM_05c`, `GM_06`, `GM_07`, `GM_08` and
# `GM_08b`, the fitted seasonal curves themselves in `GM_05d`, and the
# figures `GM_F03` to `GM_F06b`, and the on-structure fit's own per-epoch
# training and validation MAE in `GM_05e`.

# %% [markdown]
# ### Regressor-set frames and the held-out split
#
# Each regressor set's block is joined with the target and the conditional-
# seasonality weight columns into one model-ready frame per set
# (`prediction.regressor_set_frames`); rows a set cannot cover are dropped
# rather than filled. The on-structure set's frame is further split into a
# training head and a held-out tail, and its changepoints placed on that
# head's own covered time — both reused by the trend-regularisation sweep
# and the conditional-daily-term comparison below.

# %%
def_frames = prediction.regressor_set_frames(
    sets, frame['y'], roles=('tair', 'rh', 'sr'), weight_curve=WEIGHT_CURVE)

split_at = int(len(def_frames['str']) * (1 - VALID_P))
train_str = def_frames['str'].iloc[:split_at]
valid_str = def_frames['str'].iloc[split_at:]
changepoints_str = prediction.covered_changepoints(train_str.index, N_CHANGEPOINTS)

# %% [markdown]
# ### Trend regularisation swept on the held-out tail
#
# Every candidate in `TREND_REG_CANDIDATES` is fitted once on the
# on-structure set's training head and scored on its held-out tail; the
# candidate of lowest held-out MAE becomes `TREND_REG` for every fit that
# follows, and every candidate's score is written to `GM_05c` so the margin
# behind the winner is on record rather than assumed.

# %%
sweep = prediction.sweep_trend_reg(
    train_str, valid_str, TREND_REG_CANDIDATES, ('tair', 'rh', 'sr'),
    n_jobs=N_JOBS, epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
    changepoints=changepoints_str, n_changepoints=N_CHANGEPOINTS,
    changepoints_range=CHANGEPOINTS_RANGE, yearly_order=YEARLY_ORDER,
    daily_order=DAILY_ORDER, quantiles=QUANTILES, seed=SEED,
    learning_rate=LEARNING_RATE)
display(sweep)
chosen = float(sweep.loc[sweep['chosen'], 'trend_reg'].iloc[0])
TREND_REG = TREND_REG if TREND_REG is not None else chosen
print('sweep-chosen trend_reg:', chosen)
print('TREND_REG:', TREND_REG)
sweep.to_csv(OUTPUT_DIR / 'GM_05c_trend_reg_sweep.csv', index=False)
tables.write_table(sweep, str(OUTPUT_DIR / 'GM_05c_body.tex'),
                   [('trend_reg', '.1f'), ('mae_val', '.3f'),
                    ('chosen', tables.yes_no)])

# %% [markdown]
# ### The conditional daily term against the plain one
#
# Same held-out tail, now holding the trend regularisation fixed at
# `TREND_REG`: one fit with no conditional seasonality and one with the
# summer/winter daily terms. `prediction.compare_daily_terms` compares them
# by held-out MAE and by the variance share the daily term(s) carry, and
# `CONDITIONS` — the dict every attribution fit below is given — is set from
# whichever the comparison, reported in `GM_05b`, prefers.

# %%
conditional_test, keep = prediction.compare_daily_terms(
    train_str, valid_str, ('tair', 'rh', 'sr'),
    {'daily_summer': 'summer_w', 'daily_winter': 'winter_w'},
    CONDITIONAL_KEEP_MIN_SHARE, epochs=EPOCHS, freq=NATIVE_FREQ,
    growth='linear', changepoints=changepoints_str, n_changepoints=N_CHANGEPOINTS,
    changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
    yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER, quantiles=QUANTILES,
    seed=SEED, learning_rate=LEARNING_RATE)
display(conditional_test)
keep_conditional = CONDITIONAL_DAILY and keep
CONDITIONS = ({'daily_summer': 'summer_w', 'daily_winter': 'winter_w'}
              if keep_conditional else None)
print('conditional daily term kept:', keep_conditional)
conditional_test.to_csv(OUTPUT_DIR / 'GM_05b_conditional_test.csv', index=False)
tables.write_table(conditional_test, str(OUTPUT_DIR / 'GM_05b_body.tex'),
                   [('daily_term', tables.texttt), ('mae_val', '.3f'),
                    ('daily_share', lambda v: tables.percent(v, decimals=2))])

# %% [markdown]
# ### Attribution fit per set
#
# One `prediction.attribution_fits` call fits Model A once per regressor
# set, on that set's full training window, with `TREND_REG` and `CONDITIONS`
# now fixed; each fit's component variance shares, learned regressor gains
# (beside Study 03's own measurements) and residual Ljung-Box diagnostics
# are concatenated across sets and written to `GM_05`, `GM_06` and `GM_08`.
# `GM_06`'s report body omits the `r2` column NeuralProphet's own linear
# additive future regressor carries in the CSV: the fit of a learned gain
# against the component it produces is one by construction, so nine
# identical `1.000`s would say nothing the CSV does not already record.

# %%
fits, shares, gains, diagnostics = prediction.attribution_fits(
    def_frames, ('tair', 'rh', 'sr'), VALID_P, N_CHANGEPOINTS,
    study03_gains=STUDY03_GAINS, diagnostic_lags=(1, 72, 216),
    weight_curve=WEIGHT_CURVE, n_jobs=N_JOBS, epochs=EPOCHS, freq=NATIVE_FREQ,
    growth='linear', changepoints_range=CHANGEPOINTS_RANGE,
    trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
    conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
    learning_rate=LEARNING_RATE)
models_a = {n: (f['model'], f['train'], f['changepoints']) for n, f in fits.items()}
components_a = {n: f['components'] for n, f in fits.items()}

for table, number, name in ((shares, '05', 'component_shares'),
                            (gains, '06', 'learned_gains'),
                            (diagnostics, '08', 'residual_diagnostics')):
    display(table)
    table.to_csv(OUTPUT_DIR / f'GM_{number}_{name}.csv', index=False)
tables.write_table(shares, str(OUTPUT_DIR / 'GM_05_body.tex'),
                   [('set', tables.texttt), ('component', tables.texttt),
                    ('share', tables.percent), ('peak_to_peak', ',.1f')])
tables.write_table(gains, str(OUTPUT_DIR / 'GM_06_body.tex'),
                   [('set', tables.texttt), ('regressor', tables.texttt),
                    ('gain', '.3f'), ('study03_gain', '.3f')])
tables.write_table(diagnostics, str(OUTPUT_DIR / 'GM_08_body.tex'),
                   [('set', tables.texttt), ('lag', 'd'), ('lb_pvalue', '.3g'),
                    ('std', '.2f'), ('mad', '.2f')])

# %% [markdown]
# ### Native diagnostics, on-structure set only
#
# NeuralProphet's own forecast, component and parameter plots, run once on
# the on-structure fit as a diagnostic (D14) — the report's own figures
# below redraw the same content in the project's style. Each plot is
# requested in NeuralProphet's plain `'plotly'` backend, which returns the
# figure rather than showing it, and rendered inline through
# `viz.show_static`: the alternative, NeuralProphet's `'plotly-static'`
# backend, embeds every point of the whole record as inline SVG, tens of
# megabytes per figure at this record's length and cadence, where a PNG of
# the same picture is a few hundred kilobytes.

# %%
model, train, _ = models_a['str']
forecast_native = model.predict(
    prediction._model_frame(train, ('tair', 'rh', 'sr')
                            + (tuple(CONDITIONS.values()) if CONDITIONS else ())),
    decompose=True)
for native in (model.plot(forecast_native, plotting_backend='plotly'),
              model.plot_components(forecast_native, plotting_backend='plotly'),
              model.plot_parameters(plotting_backend='plotly')):
    viz.show_static(native)

# %% [markdown]
# ### Fold stability
#
# Component stability the NeuralProphet way: `prediction.fold_stability`
# takes each set's already-fitted model's own `crossvalidation_split_df`
# fold boundaries, refits from scratch on each fold's training slice, and
# reports the on-structure air-temperature gain, the trend rate and the
# yearly peak-to-peak per set and fold in `GM_07` — a component that changes
# sign or order of magnitude between folds is noise the full-window fit
# happened to land on, not a finding (§4.1).

# %%
stability = prediction.fold_stability(
    fits, def_frames, ('tair', 'rh', 'sr'), N_CHANGEPOINTS, CV_FOLDS,
    CV_FOLD_PCT, CV_FOLD_OVERLAP_PCT, NATIVE_FREQ, gain_regressor='tair',
    n_jobs=N_JOBS, epochs=EPOCHS, growth='linear',
    changepoints_range=CHANGEPOINTS_RANGE,
    trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
    conditional_seasonality=CONDITIONS, seed=SEED, learning_rate=LEARNING_RATE)
display(stability)
stability.to_csv(OUTPUT_DIR / 'GM_07_component_stability.csv', index=False)
tables.write_table(stability, str(OUTPUT_DIR / 'GM_07_body.tex'),
                   [('set', tables.texttt), ('fold', 'd'), ('tair_gain', '.3f'),
                    ('yearly_peak_to_peak', '.1f'), ('trend_rate', '.1f'),
                    ('mae_val', '.2f')])

# %% [markdown]
# ### What periodic structure the residual still carries
#
# Model A's specification is additive and does not include autoregression
# (§2.4), so whatever the trend, the annual term and the three regressors
# left unexplained is free to carry its own periodic structure — the
# spec's §4.1 asks this question of the fitted residual the same way
# `GM_04` already asked it of the raw target and the plain-regression
# residual in Movement 1b: a `prediction.period_scan` per regressor set,
# over the same short and long bands of `PERIOD_SCAN_BANDS`, ranked and
# written to `GM_08b`.

# %%
residual_scans = pd.concat(
    [prediction.period_scan(components_a[name]['residual'], min_days=lo, max_days=hi,
                            n_periods=PERIOD_SCAN_N, spacing='log', top=5)
     .assign(set=name, band=band_name)
     for name in components_a for band_name, (lo, hi) in PERIOD_SCAN_BANDS.items()],
    ignore_index=True)
display(residual_scans)
residual_scans.to_csv(OUTPUT_DIR / 'GM_08b_residual_periods.csv', index=False)
tables.write_table(residual_scans, str(OUTPUT_DIR / 'GM_08b_body.tex'),
                   [('set', tables.texttt), ('band', tables.texttt), ('rank', 'd'),
                    ('period_days', ',.4f'), ('power', '.4f')])

# %% [markdown]
# ### Trend rates and the report's own figures
#
# The on-structure fit's own trend, segment rates and seasonal curves,
# redrawn in the project's figure style rather than NeuralProphet's native
# one: fit metrics (`GM_F03`, backed by the per-epoch training and
# validation MAE themselves in `GM_05e`), the trend and its per-segment
# rates (`GM_04d`, `GM_F04`, broken across any gap longer than one native
# step), the yearly and daily curves (`GM_F05`, backed by the fitted values
# themselves in `GM_05d`), the full decomposition stack (`GM_F06`), and the
# learned gains beside Study 03's own measurements (`GM_F06b`).

# %%
model_str, train_str_fit, changepoints_str = models_a['str']
figures.plot_fit_metrics(model_str.fit_metrics_, title='Model A · on-structure set',
                         save_path=str(OUTPUT_DIR), filename='GM_F03_fit_metrics')
model_str.fit_metrics_.to_csv(OUTPUT_DIR / 'GM_05e_fit_metrics.csv', index=True)
trend, rates = prediction.trend_parameters(model_str, train_str_fit, changepoints_str,
                                           regressors=('tair', 'rh', 'sr'))
rates.to_csv(OUTPUT_DIR / 'GM_04d_trend_rates.csv', index=False)
tables.write_table(rates, str(OUTPUT_DIR / 'GM_04d_body.tex'),
                   [(tables.date_cell('start'), None), (tables.date_cell('end'), None),
                    ('rate_mdeg_per_year', '.2f')])
figures.plot_trend_parameters(trend, rates, changepoints_str, freq=NATIVE_FREQ,
                              title='Trend on covered time',
                              save_path=str(OUTPUT_DIR), filename='GM_F04_trend')
curves = prediction.seasonal_parameters(
    model_str, list(SEASONAL_CURVE_DATES) if CONDITIONS else [SEASONAL_CURVE_DATES[0]],
    freq=NATIVE_FREQ, conditions=CONDITIONS, regressors=('tair', 'rh', 'sr'))
curves.to_csv(OUTPUT_DIR / 'GM_05d_seasonal_curves.csv', index=False)
figures.plot_seasonal_parameters(curves, title='Yearly and daily terms',
                                 save_path=str(OUTPUT_DIR), filename='GM_F05_seasonality')
figures.plot_decomposition_stack(components_a['str'], freq=NATIVE_FREQ,
                                 title='Decomposition, on-structure set',
                                 save_path=str(OUTPUT_DIR), filename='GM_F06_decomposition')
figures.plot_regressor_gains(gains, title='Learned gains against Study 03',
                             save_path=str(OUTPUT_DIR), filename='GM_F06b_gains')

# %% [markdown]
# ## Movement 2b · What the wall temperature and the pyranometer buy
#
# The current era only (D4). The same specification, refitted rung by rung
# on a matched window; each rung reports its held-out error and the paired
# block-bootstrap increment over the rung below. The movement writes the
# table `GM_16` and the figure `GM_F14`.

# %% [markdown]
# ### The ladder's window and its four rungs
#
# `prediction.ladder_frame` cuts the record to the current era and adds the
# wall probe, the delayed on-structure pyranometer, the probe's
# thermal-inertia and lead variants, and the conditional-seasonality
# weights. `prediction.channel_ladder` then fits `LADDER_RUNGS` in order on
# a matched held-out split, one specification per rung, gated to the same
# rows on every rung — the window where the pyranometer and the probe are
# both present — so that a rung's held-out error differs from the rung
# below's in its added channel alone, never in a training window of a
# different size (D4). It pairs each rung's held-out error against both
# the rung immediately below it and the first rung directly, with the same
# block bootstrap Study 04 uses for its own route comparisons.

# %%
current = prediction.ladder_frame(
    frame, sensor, CURRENT_ERA_START, twall_tau_h=TWALL_TAU_H,
    twall_lead_h=TWALL_LEAD_H, radiation_delay_h=RADIATION_DELAY_H,
    freq=NATIVE_FREQ, weight_curve=WEIGHT_CURVE)
ladder, ladder_errors = prediction.channel_ladder(
    current, LADDER_RUNGS, VALID_P, LADDER_N_CHANGEPOINTS,
    block_hours=LADDER_BOOTSTRAP_BLOCK_HOURS,
    repetitions=LADDER_BOOTSTRAP_REPETITIONS, seed=SEED, n_jobs=N_JOBS,
    epochs=EPOCHS, freq=NATIVE_FREQ, growth='linear',
    changepoints_range=CHANGEPOINTS_RANGE,
    trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
    conditional_seasonality=CONDITIONS, quantiles=QUANTILES,
    learning_rate=LEARNING_RATE)
display(ladder)

# %% [markdown]
# ### The ladder table and figure
#
# `GM_16` records, per rung, the matched block's size, the held-out mean
# absolute error, interval coverage, the newest regressor's learned gain
# and the fit's residual lag-1 autocorrelation, alongside two paired
# skills: over the rung immediately below (the chain) and over the first
# rung directly, each with its bootstrap interval — the second reads a
# rung's total gain over the ladder's starting specification without
# compounding it through every rung in between. `GM_F14` draws the
# held-out MAE and the chain skill, one bar or marker per rung.

# %%
ladder.to_csv(OUTPUT_DIR / 'GM_16_current_era_ladder.csv', index=False)
tables.write_table(ladder, str(OUTPUT_DIR / 'GM_16_body.tex'),
                   [('rung', tables.texttt), ('rows', ',d'), ('mae_val', '.2f'),
                    ('skill', '.3f'), ('skill_q05', '.3f'), ('skill_q95', '.3f'),
                    ('skill_vs_first', '.3f'), ('skill_vs_first_q05', '.3f'),
                    ('skill_vs_first_q95', '.3f'), ('coverage', tables.percent),
                    ('gain_last', '.3f'), ('residual_r1', '.3f')])
figures.plot_ladder(ladder, title='What each on-structure channel buys',
                    save_path=str(OUTPUT_DIR), filename='GM_F14_ladder')

# %% [markdown]
# ## Movement 3 · Is this reading the expected one?
#
# A walk-forward expectation, refitted every `REFIT_EVERY` on the trailing
# `TRAIN_WINDOW` with the trend on (D5), and a conformal interval calibrated
# on the previous `CONFORMAL_CALIBRATION_WINDOW` of out-of-sample residuals
# (D8). Scored per set and per days since refit. The movement writes the
# table `GM_09` and the figure `GM_F08`.

# %% [markdown]
# ### The rolling expectation, per regressor set
#
# `prediction.rolling_nowcast` refits on the `REFIT_EVERY` schedule inside
# the trailing `TRAIN_WINDOW`, with `changepoints_per_window=True` so that
# no window's trend changepoints fall inside that window's own outages
# (Task 3.1). `prediction.rolling_conformal` then recalibrates each row's
# `q05`/`q95` against the rolling output's own out-of-sample residuals in
# the preceding `CONFORMAL_CALIBRATION_WINDOW`, one calibration set per
# refit origin rather than a single split fixed for the whole record (D8).
# Each row is labelled by its regressor set and by which `STALENESS_EDGES_D`
# bin its own `staleness_d` falls into.

# %%
rolling_sets = {}
for name, block in def_frames.items():
    rolling = prediction.rolling_nowcast(
        block, regressors=('tair', 'rh', 'sr'), refit_every=REFIT_EVERY,
        min_train=MIN_TRAIN, train_window=TRAIN_WINDOW,
        changepoints_per_window=True, freq=NATIVE_FREQ, n_jobs=N_JOBS,
        epochs=EPOCHS, growth='linear', n_changepoints=N_CHANGEPOINTS,
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
        conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
        learning_rate=LEARNING_RATE)
    rolling = prediction.rolling_conformal(
        rolling, alpha=CONFORMAL_ALPHA, window=CONFORMAL_CALIBRATION_WINDOW,
        method=CONFORMAL_METHOD)
    rolling['set'] = name
    rolling['staleness'] = pd.cut(rolling['staleness_d'], STALENESS_EDGES_D,
                                  labels=['0-7 d', '8-14 d', '15-21 d', '22-30 d'])
    rolling_sets[name] = rolling
    print(f'{name}: {len(rolling):,} rows from {rolling["origin"].nunique()} refits')

# %% [markdown]
# ### Scored by set and by staleness
#
# `prediction.score_predictions` pools every set's rolling predictions and
# scores each set-by-staleness group on its own out-of-sample rows: mean
# absolute error, bias, the interval's coverage and median width against
# the nominal `CONFORMAL_ALPHA`, and the Winkler interval score. `GM_09`
# records the table.

# %%
all_rolling = pd.concat(rolling_sets.values(), ignore_index=True)
nowcast = prediction.score_predictions(
    all_rolling.dropna(subset=['q05', 'q95']), ['set', 'staleness'],
    alpha=CONFORMAL_ALPHA)
display(nowcast)
nowcast.to_csv(OUTPUT_DIR / 'GM_09_nowcast_metrics.csv', index=False)
tables.write_table(nowcast, str(OUTPUT_DIR / 'GM_09_body.tex'),
                   [('set', tables.texttt), ('staleness', tables.texttt), ('n', ',d'),
                    ('mae', '.2f'), ('rmse', '.2f'), ('bias', '.2f'),
                    ('coverage_q05_q95', tables.percent), ('width_q05_q95', '.1f'),
                    ('interval_score', '.1f')])

# %% [markdown]
# ### Observed against expected, on-structure set, December 2025
#
# `GM_F08`: the on-structure rolling expectation and its conformal band
# against the measured record, drawn over the first full calendar month
# after the archive's 2025 outage (26 September – 12 October 2025) whose
# on-structure regressors are actually complete. November 2025 still
# carries a residual on-structure solar-radiation gap the outage table does
# not list separately — 82 % of `n_sr_ok` missing that month — which
# `regressor_set_frames` drops rows for outright, so it is skipped in
# favour of December 2025, whose three roles are essentially complete
# (under 0.5 % missing each).

# %%
view = rolling_sets['str'].set_index('ds').loc['2025-12-01':'2026-01-01']
figures.plot_prediction_band(
    view['y'], view['yhat'], view['q05'], view['q95'], freq=NATIVE_FREQ,
    title='Observed against expected, on-structure set, December 2025',
    save_path=str(OUTPUT_DIR), filename='GM_F08_observed_expected')

# %% [markdown]
# ### Native conformal diagnostic, on-structure set only
#
# NeuralProphet's own `conformal_predict`/`conformal_plot`, run once on the
# on-structure fit as a diagnostic counterpart to `GM_F08` (D14) rather than
# a second scored result — an 80/20 split of that fit's own training frame
# stands in for the calibration and evaluation data respectively, distinct
# from the rolling calibration `GM_09` and `GM_F08` use. As in Movement 2's
# native diagnostics, the plot is requested in NeuralProphet's plain
# `'plotly'` backend and rendered to PNG inline through `viz.show_static`,
# never the whole-record-as-SVG `'plotly-static'` backend. `conformal_plot`
# reads every retained interval width, not only the one `'cqr'` keeps by
# default, so `conformal_predict` is called with `show_all_PI=True`.

# %%
model_str, train_str_fit, _ = models_a['str']
split = int(len(train_str_fit) * 0.8)
passthrough = ('tair', 'rh', 'sr') + (tuple(CONDITIONS.values()) if CONDITIONS else ())
native = model_str.conformal_predict(
    prediction._model_frame(def_frames['str'].iloc[len(train_str_fit):], passthrough),
    calibration_df=prediction._model_frame(train_str_fit.iloc[split:], passthrough),
    alpha=CONFORMAL_ALPHA, method=CONFORMAL_METHOD, show_all_PI=True)
native_fig = model_str.conformal_plot(native, plotting_backend='plotly')
if native_fig is not None:
    viz.show_static(native_fig)
else:
    print('conformal_plot returned None under the plotly backend; '
         'native diagnostic skipped rather than embedding SVG.')

# %% [markdown]
# ## Movement 4 · Does the wall answer with a delay the 20-minute grid can resolve?
#
# The same specification as Model A's on-structure fit, except that air
# temperature and radiation (`LAGGED_REGRESSORS`) enter as lagged
# regressors over `LAGGED_N_LAGS` slots of the native 20-minute grid rather
# than as contemporaneous ones (D9). The weight the fit learns at every lag
# is the impulse response itself; its first moment and its one-pole fit are
# read back and set beside `STUDY03_OPERATOR`, the delay-and-time-constant
# operator Study 03 measured by scanning rather than by learning. The
# movement writes the weights and summary tables in `GM_10` and the figure
# `GM_F07`. Phase 0's `test_neuralprophet_capabilities.
# TestLaggedRegressorWithoutAutoregression` already confirmed that
# NeuralProphet 0.8.0 carries lagged regressors at `n_lags=0`, so
# `MODEL_B_FALLBACK_FREQ`'s hourly fallback is not taken and this movement
# runs at the native grid throughout.

# %% [markdown]
# ### The lagged-regressor fit
#
# One nowcast fit on the on-structure set, sharing every specification
# choice already fixed for Model A (`N_CHANGEPOINTS`, `CHANGEPOINTS_RANGE`,
# `TREND_REG`, `YEARLY_ORDER`, `DAILY_ORDER`, `CONDITIONS`): the only
# difference is that air temperature and radiation are registered through
# `lagged_regressors` instead of `regressors`, carrying `LAGGED_N_LAGS`
# slots of their own recent history into the fit, while relative humidity
# stays a contemporaneous regressor exactly as in Model A.
# `prediction.lagged_regressor_weights` reads the fitted weight at every
# lag straight back from the model, called with `physical=True` so that
# every weight is rescaled from the model's own internal normalisation
# into millidegrees per unit of the driver — the units `STUDY03_GAINS`
# and Study 03's operator are both stated in — rather than left in
# standard deviations of the (also normalised) target per unit of
# whatever normalisation the driver itself happened to receive.
# `prediction.impulse_response_summary` reduces that physical-units table
# to one row per driver — gain, delay and time constant — with Study 03's
# own operator attached alongside for the comparison the movement exists
# to make; delay and time constant are shape quantities read off the
# response's own timing, not its scale, so they are identical whichever
# units the weights themselves are read in.

# %%
block_b = def_frames['str']
slots_per_hour = int(pd.Timedelta(hours=1) / pd.Timedelta(NATIVE_FREQ))
model_b, _ = prediction.neuralprophet_backtest(
    block_b, block_b, regressors=('rh',), task='nowcast', epochs=EPOCHS,
    freq=NATIVE_FREQ, growth='linear',
    changepoints=prediction.covered_changepoints(block_b.index, N_CHANGEPOINTS),
    n_changepoints=N_CHANGEPOINTS, changepoints_range=CHANGEPOINTS_RANGE,
    trend_reg=TREND_REG, yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
    conditional_seasonality=CONDITIONS, quantiles=(), seed=SEED,
    learning_rate=LEARNING_RATE, lagged_regressors=LAGGED_REGRESSORS,
    lagged_n_lags=LAGGED_N_LAGS, lagged_regularization=LAGGED_REG)
weights_b = prediction.lagged_regressor_weights(model_b, physical=True)
summary_b = prediction.impulse_response_summary(weights_b, dt_hours=1.0 / slots_per_hour)
summary_b['study03_delay_h'] = [STUDY03_OPERATOR[r]['delay_h'] for r in summary_b['regressor']]
summary_b['study03_tau_h'] = [STUDY03_OPERATOR[r]['tau_h'] for r in summary_b['regressor']]

# %% [markdown]
# ### `GM_10` · the learned response beside Study 03's operator
#
# One row per driver: the learned gain, delay and time constant, and
# Study 03's own delay and time constant for the same driver, so the
# margin between what was imposed and what the model found is on record
# rather than merely visible in the figure below.

# %%
display(summary_b)
weights_b.to_csv(OUTPUT_DIR / 'GM_10_impulse_response_weights.csv', index=False)
summary_b.to_csv(OUTPUT_DIR / 'GM_10_impulse_response.csv', index=False)
tables.write_table(summary_b, str(OUTPUT_DIR / 'GM_10_body.tex'),
                   [('regressor', tables.texttt), ('gain', '.3f'), ('delay_h', '.2f'),
                    ('tau_h', '.2f'), ('r2_onepole', '.3f'),
                    ('study03_delay_h', '.1f'), ('study03_tau_h', '.1f')])

# %% [markdown]
# ### `GM_F07` and the native diagnostic
#
# The learned weight per lag in millidegrees per unit of the driver, one
# panel per driver, with Study 03's operator overlaid as a gain-matched
# one-pole response. Beside it, NeuralProphet's own lagged-regressor
# parameter plot runs once as a diagnostic counterpart (D14) — in the
# model's own internal normalisation rather than physical units, since
# that plot reads the raw tensor directly — requested in the plain
# `'plotly'` backend and rendered to PNG inline through `viz.show_static`
# rather than embedded as SVG; if that backend returns no figure, the
# diagnostic is skipped and said so in print rather than falling back to
# SVG.

# %%
figures.plot_impulse_response(weights_b, summary_b, reference=STUDY03_OPERATOR,
                              dt_hours=1.0 / slots_per_hour,
                              unit_label='Weight [mdeg per unit]',
                              title='Learned impulse response, on-structure set',
                              save_path=str(OUTPUT_DIR), filename='GM_F07_impulse_response')
native_params = model_b.plot_parameters(components=['lagged_regressors'],
                                        plotting_backend='plotly')
if native_params is not None:
    viz.show_static(native_params)
else:
    print('plot_parameters returned None under the plotly backend; '
         'native diagnostic skipped rather than embedding SVG.')

# %% [markdown]
# ## Movement 5 · What departure does each chart catch?
#
# The rolling residual of the on-structure set (`rolling_sets['str']`) is
# charted three ways (D10): prewhitened innovations at twenty minutes,
# built for a sudden departure; the amplitude and phase of its daily
# cycle, built for a changed daily response; and its daily mean, built for
# drift. Reference statistics are estimated on the fixed window
# `REFERENCE_START` to `REFERENCE_END` — the rolling residual's own
# history begins partway through the record, so this window is shorter
# than first planned, and the report states why. Each chart's control
# limit is swept to its own false-alarm budget; the fast chart's alarms
# are cross-checked against the on-structure environment and supply
# channels; and detectability is measured per damage mechanism on the
# chart and statistic each one is actually scored on — every mechanism on
# every chart, not only its own — with injections sized by the wall's own
# measured daily response (D11). Writes `GM_11` and `GM_12` (the alarm
# episodes and the tuned run lengths), `GM_13` (the full detectability
# sweep) and `GM_13b` (the detection threshold read off each mechanism's
# own chart), and `GM_F09`–`GM_F11` and `GM_F13_amplitude`, `GM_F13_phase`,
# `GM_F13_drift`, `GM_F13_step`.

# %% [markdown]
# ### The charted series
#
# `monitoring.chart_series` reduces the on-structure rolling residual to
# the four series the fast, daily and slow charts are actually built on —
# prewhitened innovations, the daily harmonic's amplitude and phase, and
# the daily mean. The per-chart budget, smoothing constant and coincidence
# window each carries come from the parameter cell, attached alongside its
# series so that `charts` states everything one chart needs in one place.

# %%
rolling_str = rolling_sets['str'].set_index('ds').sort_index()
residual = (rolling_str['y'] - rolling_str['yhat']).asfreq(NATIVE_FREQ)

series_by_name, phi = monitoring.chart_series(
    residual, NATIVE_FREQ, DAILY_HARMONIC_MIN_SLOTS, REFERENCE_START, REFERENCE_END)
chart_specs = {
    'fast': dict(freq=NATIVE_FREQ, budget=BUDGET_FAST_DAYS, lam=EWMA_LAMBDA,
                joint=JOINT_WINDOW),
    'daily_amplitude': dict(freq='1D', budget=BUDGET_DAILY_DAYS,
                            lam=EWMA_LAMBDA_DAILY, joint=JOINT_WINDOW_DAILY),
    'daily_phase': dict(freq='1D', budget=BUDGET_DAILY_DAYS,
                        lam=EWMA_LAMBDA_DAILY, joint=JOINT_WINDOW_DAILY),
    'slow': dict(freq='1D', budget=BUDGET_SLOW_DAYS, lam=EWMA_LAMBDA_SLOW,
                joint=JOINT_WINDOW_DAILY),
}
charts = {name: {**spec, 'series': series_by_name[name]}
         for name, spec in chart_specs.items()}
print(f'phi = {phi:.4f}; reference-window residual sd '
     f'{residual.loc[REFERENCE_START:REFERENCE_END].std():.2f}, '
     f'innovation sd {series_by_name["fast"].loc[REFERENCE_START:REFERENCE_END].std():.2f}')

# %% [markdown]
# ### Tuning each chart to its false-alarm budget, and running it
#
# `monitoring.tune_limit_to_budget` sweeps `LIMIT_CANDIDATES` on the
# reference stretch alone and returns the smallest control limit whose
# average run length meets that chart's budget, with the full sweep table
# alongside it; `monitoring.run_chart` then runs the EWMA and CUSUM charts
# and their joint alarm on the monitored stretch at that limit and
# collapses the alarm into episodes. `GM_12` records, for every chart, the
# limit chosen and the run length it actually achieves against the budget
# it was tuned to.

# %%
tuned = {}
for name, spec in charts.items():
    reference = monitoring.reference_stats(
        spec['series'], start=REFERENCE_START, end=REFERENCE_END)
    L, sweep = monitoring.tune_limit_to_budget(
        spec['series'], REFERENCE_START, REFERENCE_END, spec['budget'],
        LIMIT_CANDIDATES, spec['lam'], CUSUM_K, CUSUM_H, spec['joint'], spec['freq'])
    run = monitoring.run_chart(
        spec['series'], reference, L, spec['lam'], CUSUM_K, CUSUM_H,
        spec['joint'], MONITORED_START, spec['freq'])
    tuned[name] = {'reference': reference, 'L': L, 'sweep': sweep, **run}

runs = pd.DataFrame([
    {'chart': name, 'L': tuned[name]['L'], 'budget_days': charts[name]['budget'],
     'achieved_arl_days': tuned[name]['sweep'].loc[
         tuned[name]['sweep']['L'] == tuned[name]['L'], 'arl_days'].item(),
     'episodes_in_reference': tuned[name]['sweep'].loc[
         tuned[name]['sweep']['L'] == tuned[name]['L'], 'n_episodes'].item()}
    for name in charts])
display(runs)
runs.to_csv(OUTPUT_DIR / 'GM_12_run_lengths.csv', index=False)
tables.write_table(runs, str(OUTPUT_DIR / 'GM_12_body.tex'),
                   [('chart', tables.texttt), ('L', '.2f'), ('budget_days', '.0f'),
                    ('achieved_arl_days', '.0f'), ('episodes_in_reference', ',d')])

# %% [markdown]
# ### Attribution of the fast chart's alarms
#
# `monitoring.channel_coincidence` reduces air temperature, relative
# humidity and supply voltage — the raw on-structure channels, from
# `sensor`, before any dust-gap filling — to their departure from a
# centred rolling median at `ATTRIBUTION_WINDOW`, scaled on the reference
# window by `ATTRIBUTION_THRESHOLD`, and labels every fast-chart alarm slot
# by whichever of them was also in excursion. The window is two hours, not
# the function's own twenty-four: at a full day the "departure from the
# median" on air temperature is the diurnal cycle itself, whose own swing
# swamps the threshold and left every episode of the first run
# `unattributed` — the coincidence test is built to catch a twenty-minute
# swing against its own local background, not a cycle the median should
# already track out. `monitoring.attribute_episodes` reduces those slot
# labels to one attribution per episode: the mode of the labels falling
# inside its span. `GM_11` carries every chart's episodes, with the
# attribution filled in for the fast chart and the table's missing marker
# elsewhere — the daily and slow charts are not cross-checked against
# these channels, since a swing over a day or a year is not what a
# twenty-minute coincidence test is built to catch.

# %%
attribution_channels = sensor[[f'{c}_str' for c in ATTRIBUTION_CHANNELS]].rename(
    columns={f'{c}_str': c for c in ATTRIBUTION_CHANNELS}).reindex(residual.index)
labels = monitoring.channel_coincidence(
    tuned['fast']['joint'], attribution_channels,
    scale_start=REFERENCE_START, scale_end=REFERENCE_END,
    window=ATTRIBUTION_WINDOW, threshold=ATTRIBUTION_THRESHOLD)
fast_episodes = monitoring.attribute_episodes(
    tuned['fast']['episodes'], labels).assign(chart='fast')
other_episodes = pd.concat(
    [tuned[name]['episodes'].assign(chart=name) for name in charts if name != 'fast'],
    ignore_index=True)
episodes = pd.concat([fast_episodes, other_episodes], ignore_index=True)
display(episodes)
episodes.to_csv(OUTPUT_DIR / 'GM_11_alarm_episodes.csv', index=False)
tables.write_table(episodes, str(OUTPUT_DIR / 'GM_11_body.tex'),
                   [('chart', tables.texttt), (tables.date_cell('start'), None),
                    (tables.date_cell('end'), None), ('duration_h', ',.0f'),
                    ('mean_z', '.2f'), ('attribution', tables.texttt)])

# %% [markdown]
# ### `GM_F09`–`GM_F11`: the three charts
#
# The fast and slow charts reuse `figures.plot_control_chart`, drawing the
# EWMA and CUSUM statistics respectively against their tuned limits with
# every alarm episode shaded; the daily chart reuses the same colours and
# shading through `figures.plot_daily_harmonic_chart`, stacking the
# amplitude and phase EWMA panels on one clock.

# %%
figures.plot_control_chart(
    tuned['fast']['ewma'], statistic='ewma', episodes=tuned['fast']['episodes'],
    freq=NATIVE_FREQ, title='Fast chart: prewhitened innovations',
    save_path=str(OUTPUT_DIR), filename='GM_F09_fast_chart')
figures.plot_daily_harmonic_chart(
    tuned['daily_amplitude']['ewma'], tuned['daily_phase']['ewma'],
    episodes=pd.concat([tuned['daily_amplitude']['episodes'],
                        tuned['daily_phase']['episodes']], ignore_index=True),
    title='Daily chart: amplitude and phase of the daily cycle',
    save_path=str(OUTPUT_DIR), filename='GM_F10_daily_chart')
figures.plot_control_chart(
    tuned['slow']['cusum'], statistic='cusum_high', episodes=tuned['slow']['episodes'],
    freq='1D', title='Slow chart: daily-mean residual',
    save_path=str(OUTPUT_DIR), filename='GM_F11_slow_chart')

# %% [markdown]
# ### Detectability per mechanism, on every chart
#
# `monitoring.daily_response_amplitude` reads the wall's own fitted daily
# response — the air-temperature component plus every daily seasonal term
# — on the injection dates, so the phase mechanism's timing shifts
# (`DETECT_PHASE_SHIFTS_H`) are converted to a residual amplitude by
# `monitoring.phase_shift_amplitude` against a response the model actually
# learned rather than an arbitrary figure. The drift mechanism sweeps
# `DETECT_DRIFT_HORIZONS` rather than `DETECT_DURATIONS` — a drift needs
# weeks, not hours, to accumulate into anything a chart could see —
# assembled into a per-mechanism `durations` dict by a comprehension over
# `MECHANISM_CHARTS` with the drift key replaced. `monitoring.
# detectability_by_mechanism` then sweeps each mechanism not only on its
# own chart but on every chart named in `MECHANISM_CHARTS` (`all_charts=
# True`), over the reference window's own residual — the only stretch
# known to be in control — so the report can say what a chart built for
# one mechanism does or does not catch of the other three. Before the
# sweep, the notebook prints, for each injection date, how much of the
# following twenty days the reference residual actually covers, since a
# date sitting against a gap would understate what the sweep could find.

# %%
for date in DETECT_INJECTION_DATES:
    window = residual.loc[pd.Timestamp(date):pd.Timestamp(date) + pd.Timedelta(days=20)]
    print(f'{date}: {window.notna().mean():.1%} of the following 20 days covered')

response_amplitude = monitoring.daily_response_amplitude(
    components_a['str'], DETECT_INJECTION_DATES, window=72,
    min_slots=DAILY_HARMONIC_MIN_SLOTS, driver='future_regressor_tair')
phase_magnitudes = tuple(
    monitoring.phase_shift_amplitude(response_amplitude, h) for h in DETECT_PHASE_SHIFTS_H)
detect_magnitudes = {'amplitude': DETECT_MAGNITUDES, 'phase': phase_magnitudes,
                     'drift': DETECT_DRIFT_RATES, 'step': DETECT_MAGNITUDES}
detect_durations = {name: DETECT_DURATIONS for name in MECHANISM_CHARTS}
detect_durations['drift'] = DETECT_DRIFT_HORIZONS
injection_starts = [d for d in DETECT_INJECTION_DATES
                    if REFERENCE_START <= d <= REFERENCE_END] or None

detectability = monitoring.detectability_by_mechanism(
    residual.loc[REFERENCE_START:REFERENCE_END], tuned, charts, MECHANISM_CHARTS,
    detect_magnitudes, detect_durations, NATIVE_FREQ, CUSUM_K, CUSUM_H, phi,
    DETECT_RESPONSE_WINDOW, injection_starts, DAILY_HARMONIC_MIN_SLOTS, seed=SEED,
    all_charts=True)
display(detectability)
detectability.to_csv(OUTPUT_DIR / 'GM_13_detectability.csv', index=False)
primary_detectability = detectability[detectability['primary']]
tables.write_table(primary_detectability, str(OUTPUT_DIR / 'GM_13_body.tex'),
                   [('mechanism', tables.texttt), ('chart', tables.texttt),
                    ('magnitude', '.2f'), ('duration_h', '.0f'), ('detected', '.2f'),
                    ('delay_h', '.1f')])
figures.plot_detectability(
    primary_detectability[primary_detectability['mechanism'] == 'amplitude'],
    title='Amplitude growth on the daily chart',
    save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_amplitude')
figures.plot_detectability(
    primary_detectability[primary_detectability['mechanism'] == 'phase'],
    title='Phase change on the daily chart',
    save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_phase')
figures.plot_detectability(
    primary_detectability[primary_detectability['mechanism'] == 'drift'],
    title='Drift on the slow chart',
    save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_drift')
figures.plot_detectability(
    primary_detectability[primary_detectability['mechanism'] == 'step'],
    title='Step on the fast chart',
    save_path=str(OUTPUT_DIR), filename='GM_F13_detectability_step')

# %% [markdown]
# ### `GM_13b`: the detection threshold read off each mechanism's own chart
#
# `monitoring.detection_thresholds` reduces the full sweep to one number a
# reader actually wants: at the longest horizon swept, the smallest
# magnitude a chart catches at all and the smallest it catches on every
# injection, for every mechanism-and-chart pair — not only the primary
# ones, so a chart's blindness to a mechanism it was not built for is on
# the record too.

# %%
thresholds = monitoring.detection_thresholds(detectability)
display(thresholds)
thresholds.to_csv(OUTPUT_DIR / 'GM_13b_detection_thresholds.csv', index=False)
tables.write_table(thresholds, str(OUTPUT_DIR / 'GM_13b_body.tex'),
                   [('mechanism', tables.texttt), ('chart', tables.texttt),
                    ('horizon_h', '.0f'), ('smallest_any', '.2f'),
                    ('smallest_all', '.2f'), ('delay_h_at_smallest_all', '.1f'),
                    ('primary', lambda value: tables.texttt('yes' if value else 'no'))])

# %% [markdown]
# ## Movement 6 · What happened across each outage?
#
# `prediction.outage_bridge` asks the question an outage cannot answer on
# its own: whether the missing days hid a real movement of the wall, or
# were simply a gap in an otherwise unremarkable record (D12). For each of
# the seven whole-day outages the function fits a model on everything
# before the gap, carries its expectation and its own quantile band
# through the gap on the proxies that kept recording, and compares the
# level the station reports over `OUTAGE_WINDOW_DAYS` once it resumes —
# past `OUTAGE_SETTLE_DAYS`'s restart transient — against that
# expectation. Nothing is written into the gap on either side of the fit,
# and the interval quoted through it is the fitted model's own band from
# before the outage, since no conformal calibration exists inside a gap to
# draw one from instead. Writes `GM_14` (the bridge table, every outage
# and both sets) and `GM_F12` (the station set's bridges, drawn panel by
# panel).

# %% [markdown]
# ### Bridging every outage on the station and ERA5 sets
#
# `runner=None` asks `outage_bridge` to fit its own bridge through
# `neuralprophet_backtest` at `task='nowcast'`, with changepoints kept off
# the very gap the pre-outage training data does not cover
# (`covered_changepoints`) and every other setting carried from the
# parameter cells the same way Movement 3's rolling expectation carries
# them. `MIN_TRAIN` guards an outage sitting too close to the start of a
# set's own usable history, reporting it `'no data'` without a fit rather
# than fitting on a training window this study would not otherwise trust;
# `BRIDGE_SETS` excludes the on-structure set outright rather than relying
# on that guard alone, since the on-structure package is exactly what
# every outage took down and so it has no regressor data of its own inside
# a gap to bridge with. The loop over outages inside `outage_bridge` runs
# at `N_JOBS`, one worker process per outage.

# %%
bridge_rows, bridge_paths = [], []
for name in BRIDGE_SETS:
    block = def_frames[name]
    table, paths = prediction.outage_bridge(
        None, block, OUTAGES, settle_days=OUTAGE_SETTLE_DAYS,
        window_days=OUTAGE_WINDOW_DAYS, regressors=('tair', 'rh', 'sr'),
        min_train=MIN_TRAIN, n_jobs=N_JOBS, epochs=EPOCHS, freq=NATIVE_FREQ,
        growth='linear', n_changepoints=N_CHANGEPOINTS,
        changepoints_range=CHANGEPOINTS_RANGE, trend_reg=TREND_REG,
        yearly_order=YEARLY_ORDER, daily_order=DAILY_ORDER,
        conditional_seasonality=CONDITIONS, quantiles=QUANTILES, seed=SEED,
        learning_rate=LEARNING_RATE)
    bridge_rows.append(table.assign(set=name))
    bridge_paths.append(paths.assign(set=name))
bridges = pd.concat(bridge_rows, ignore_index=True)
display(bridges)
bridges.to_csv(OUTPUT_DIR / 'GM_14_outage_bridges.csv', index=False)
tables.write_table(bridges, str(OUTPUT_DIR / 'GM_14_body.tex'),
                   [('set', tables.texttt), ('outage', 'd'),
                    (tables.date_cell('start'), None), (tables.date_cell('end'), None),
                    ('expected', '.1f'), ('observed', '.1f'), ('shift', '.1f'),
                    ('lower', '.1f'), ('upper', '.1f'), ('verdict', tables.texttt)])

# %% [markdown]
# ### `GM_F12`: the station set's bridges
#
# `figures.plot_outage_bridge` draws the first set of `BRIDGE_SETS` — the
# station set, the proxy source physically closest to the wall — one
# panel per outage, the observed target against the expected level and
# its band, with the post-resumption window the verdict is judged over
# shaded.

# %%
figures.plot_outage_bridge(bridge_paths[0], bridge_rows[0],
                           title=f'Outage bridges, {BRIDGE_SETS[0]} set',
                           save_path=str(OUTPUT_DIR), filename='GM_F12_outage_bridges')
