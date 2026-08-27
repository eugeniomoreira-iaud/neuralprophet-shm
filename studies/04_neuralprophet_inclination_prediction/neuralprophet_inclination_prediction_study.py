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
# # Study 04 · Predicting the station 02 inclination
#
# Study 01 decided which recorded values are measurements and exported the
# verdict-aware archive. Its final inclination product — compensated, anchored
# once across the whole record, cleaned of impulsive noise, and flagged wherever
# a value was interpolated rather than measured — is the only inclination series
# this study uses.
#
# The subject is one sensor at one place: the inclinometer at station 02. The
# question is operational. Given the record as it actually stands, can the next
# inclination value be predicted — from the inclination's own history, from the
# other variables the package on the structure measures beside it, or from
# external environmental proxies?
#
# The archive carries a 271-day outage ending on 20 June 2023. This study works
# only on what follows it, the stretch closest to the present and the only one
# over which the complete instrument package exists. That window is not
# continuous either, so the study proceeds in four movements:
#
# 1. Look at the window whole — every on-structure variable on one clock, with
#    the gaps left as gaps.
# 2. Inventory the gaps, classifying the missing time by duration.
# 3. Weigh the ways each class of gap might be filled.
# 4. Put the prediction question to the on-structure measurements, the external
#    proxies, or both.
#
# Movements 2 to 4 are being rebuilt to the design in
# docs/superpowers/specs/2026-08-25-study04-decomposition-and-anomaly-design.md.
# The previous experiment's cells were removed rather than held inert; its
# reusable logic survives in shmlib.prediction.

# %% [markdown]
# ## Imports and parameters
#
# All library operations live in `shmlib`. This notebook owns the data
# pointers and the choices that make this one experiment: the response channel,
# the window, and the variables drawn from the on-structure package.

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

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

from shmlib import (adc, figures, monitoring, prediction, proxies, site,
                    tables, viz)

warnings.filterwarnings('ignore')
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)
viz.apply_report_style()

# %%
# ---------------------------------------------------------------------------
# Data pointers
# ---------------------------------------------------------------------------
ARCHIVE_CSV = '../../data/interim/archive/gubbio_archive_20min.csv'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
# The archive is written every twenty minutes and this study reads it at that
# spacing, because that is the rate at which a deployed system receives a new
# reading and therefore the rate at which it must judge one. Model A works on
# this grid. Model B works hourly: measured on this record, the twenty-minute
# first difference has a lag-one autocorrelation of -0.058, the signature of
# measurement noise dominating the increment, while the hourly difference
# retains +0.349 of memory and couples to air temperature at -0.893 rather
# than -0.778. Notebook step 3 re-measures both and exports the comparison.
NATIVE_FREQ = '20min'
MODEL_FREQ_A = '20min'          # decomposition and expectation
MODEL_FREQ_B = '1h'             # forecast skill

# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------
# Study 01's final inclination product: compensated, anchored once across the
# whole record, and cleaned of impulsive noise. Read with honour_spike=True, so
# every value the cleaning interpolated is returned missing rather than as a
# measurement. No earlier or intermediate version of this channel is used.
TARGET_COLUMN = 'inc_comp_cleaned'

# No raw channel is read. Compensation is Study 01's discussion and this study
# is blind to it (D10): the compensated, cleaned channel above is the raw
# material here, and the sign contradiction of docs/raw-data-format.md section
# 7.5 stays Study 01's open question.

# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
# The archive carries a 271-day outage beginning 2022-09-23 whose last missing
# day is 2023-06-20. This study starts on the day after it and runs to the end
# of the archive: the stretch closest to the present, and the one a system
# deployed today would have to work with.
SEGMENT_START = '2023-06-21'

# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
# Air temperature and relative humidity only. Measured on this record, their
# missingness is perfectly nested inside the inclination's, so they cost no
# coverage at all: requiring them leaves the same complete rows in the same
# segments. Wall temperature and solar radiation are excluded from every model
# by design decision D11 - requiring them collapses coverage from 75.6% to
# 17.0%. They are still loaded and still drawn in the record figure below,
# because what the package records is part of what this study has to describe,
# and the reader is owed the sight of the two channels the study set aside.
PREDICTOR_COLUMNS = ('tair', 'rh')

# Battery voltage is carried as the negative control only. Study 03 established
# that it is not a silent channel - it reaches r = -0.673 in the diurnal band -
# so it marks a conservative floor rather than a zero.
CONTROL_COLUMN = 'batt'

# ---------------------------------------------------------------------------
# Model A - decomposition and expectation
# ---------------------------------------------------------------------------
# Zero autoregressive lags. With n_lags > 0 the autoregressive component absorbs
# most of the diurnal structure, and the seasonal component becomes the
# periodicity left over after it rather than the wall's thermal cycle. A
# decomposition meant to be read as physics must therefore carry no AR term.
MODEL_A_LAGS = 0

# Two fits, because one model cannot do both jobs on this record.
#
# The attribution fit carries a piecewise-linear trend: the drift is the
# structurally interesting component and does not exist under growth='off'.
# Its numbers are read in-sample, where a fitted trend is constrained by data.
MODEL_A_GROWTH = 'linear'

# The monitoring fit carries no trend at all. Measured on this record, a
# piecewise-linear trend extrapolated past its last changepoint dominates
# everything downstream: over the evaluation stretch the residual of the
# attribution fit has a standard deviation of 41.9 mdeg and reaches -194,
# against 13.8 and -43 for the same model without the trend. A control chart
# calibrated on the first would spend its entire alarm budget on the model's
# own extrapolation error. Dropping the trend leaves the drift inside the
# residual, which is the right place for it: a monitoring system should watch
# a drift, not have it subtracted away before it looks.
MODEL_A_MONITOR_GROWTH = 'off'

# Changepoints are placed at quantiles of the observed timestamps rather than
# uniformly along the axis: this window contains outages of 40, 42, 17 and 103
# days, and a changepoint inside one is constrained by no data.
MODEL_A_CHANGEPOINTS = 12

# Yearly seasonality is fitted both ways, because the comparison is worth
# reporting: the window spans 3.2 annual cycles with a 103-day hole in the last
# one, which is not obviously enough to identify an annual term, and on this
# record the annual term costs held-out accuracy rather than buying it.
MODEL_A_YEARLY_CANDIDATES = (False, True)

# The annual term is kept regardless of that comparison. Model A exists to
# attribute variation to named causes, not to minimise a forecast error, and the
# two goals disagree here. Without the term the residual carries a slow arch of
# about 75 mdeg peak to trough - the annual cycle itself - and everything
# downstream reads the residual as evidence of structural departure. A detector
# calibrated on it would spend its alarm budget on the seasons and could not see
# a millidegree-scale movement underneath. Leaving a known cause unmodelled to
# buy held-out accuracy would be buying the wrong thing.
MODEL_A_YEARLY = True

# Ljung-Box lags for the residual diagnostics: one slot, one day, three days.
RESIDUAL_LAGS = (1, 72, 216)

# Nominal miss rate of the interval, used by the Winkler interval score. It must
# match MODEL_A_QUANTILES above: change both together or the score stops being
# comparable with the coverage reported beside it.
NOWCAST_INTERVAL_ALPHA = 0.10

# The stretch drawn in the observed-against-expected figure. One month, chosen
# for readability rather than for flattery: it sits inside the evaluation period
# and carries both complete days and dropouts.
NOWCAST_VIEW = ('2025-11-01', '2025-12-01')

# The stretch drawn in the daily-cycle figure. A fortnight, short enough that
# individual cycles are legible.
COMPONENT_VIEW = ('2025-11-01', '2025-11-15')

# ---------------------------------------------------------------------------
# The rolling expectation
# ---------------------------------------------------------------------------
# How often the monitoring model is refitted, and how much history it must have
# before it is allowed to speak. A frozen fit is not a monitoring system: fitted
# once to 2025-09 and applied afterwards, this model sits 28.75 mdeg away from
# the record, and 82% of its mean square error is that constant offset rather
# than scatter. Refitting monthly holds the expectation to within the drift a
# month can accumulate, which at the measured -2.77 mdeg/yr is about 0.23 mdeg.
ROLLING_REFIT_EVERY = '30d'

# A window must hold at least one full annual cycle before it is allowed to
# speak, because the model carries an annual term and a term fitted to less than
# a cycle is unidentifiable: it can take any amplitude and phase that flatters
# the months it happens to see. At 180 days the early windows did exactly that,
# and their residuals both dragged the average and fattened the tails the
# conformal interval calibrates on.
ROLLING_MIN_TRAIN = '365d'

# Where the conformal interval is calibrated. Residual quantiles are taken at or
# before this instant and applied everywhere, so no row is ever judged against
# an interval calibrated on itself. It matches the reference window that step 7
# calibrates its control charts on, for the same reason.
CONFORMAL_CALIBRATION_END = '2025-06-01'

MODEL_A_EPOCHS = 30
MODEL_A_QUANTILES = (0.05, 0.95)
MODEL_A_SEED = 0

# Training origin: the model is fitted on everything before this instant and
# judged on everything after it. Chronological, never random.
MODEL_A_TRAIN_END = '2025-09-01'

# Minimum segment length, in slots. A segment shorter than a day cannot inform
# a daily seasonality, and contributes noise to the trend.
MODEL_A_MIN_SEGMENT = 72

# ---------------------------------------------------------------------------
# Channel maps
# ---------------------------------------------------------------------------
# The archive names the package's channels differently before and after the
# instrument installed on 2025-02-21, so each set is loaded under its own map
# and the two are joined. This is a column-naming detail and not an analytical
# split: Study 01 already compensated and anchored the inclination once across
# the whole record, and no era offset is estimated anywhere in this study.
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

# Panels of the record figure, response first, then the thermal channels, then
# the rest. Supply voltage is instrument housekeeping rather than a measurement
# of the wall or its environment, and is not drawn.
ON_STRUCTURE_COLUMNS = (
    TARGET_COLUMN, 'twall_str', 'tair_str', 'sr_str', 'rh_str')

# ---------------------------------------------------------------------------
# Gap anatomy and segmentation
# ---------------------------------------------------------------------------
# Autoregressive windows to evaluate against the record's contiguity, in hours.
# The chosen value is read off NP_03 in step 2 and set in step 6.
SURVIVAL_LAG_HOURS = (4, 8, 12, 24, 48)
SURVIVAL_FORECAST_HOURS = (8, 24)

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
# The reference window, which is also where the monitored record begins. Every
# limit drawn is a multiple of the centre and scale estimated here, so a window
# containing a departure would calibrate the detector against the very thing it
# is meant to find.
#
# It ends well before the stretch Study 01 flagged in summer 2026 - excluded as
# a precaution, since a reference window must be in control. This study makes no
# detection claim about that stretch (spec section 11.3).
#
# It starts after the rolling expectation's burn-in rather than at the first
# window that exists. Measured on this record, the residual's 30-day rolling
# mean runs -60.5, -61.0, -34.2 and -35.5 mdeg over June to September 2024 and
# then settles to +3.4 and stays inside roughly +/-20 mdeg for the remainder.
# The early windows train on barely one calendar annual cycle, holed by outages
# of 40, 42 and 17 days, so their annual term is fitted to less than one cycle -
# the same effect that forced ROLLING_MIN_TRAIN up from 180 days, still present
# at 365. Those months are excluded from the monitored record as well as from
# the reference window: an expectation wrong by 60 mdeg because it has not yet
# seen a full cycle is a fact about the model, and reporting it as a structural
# departure would be a false claim.
REFERENCE_START = '2024-10-01'
REFERENCE_END = '2025-06-01'

# EWMA smoothing and limit width. Lambda smaller reacts more slowly and finds
# smaller sustained shifts; L wider means fewer false alarms and later
# detection. L is swept over EWMA_L_CANDIDATES in step 7 and the value meeting
# the false-alarm budget replaces the one set here.
EWMA_LAMBDA = 0.05
EWMA_L = 3.0

# The limit widths swept, in standard deviations. The range runs far past the
# 2 to 5 a control-chart text would offer, and deliberately. Those widths are
# derived for independent samples; this residual has a lag-one autocorrelation
# of 0.997 at twenty minutes and still 0.98 at twenty-four hours, so a textbook
# width against it buys days between false alarms rather than months. The width
# that meets the budget is therefore found by measurement over a range wide
# enough to bracket it, and the run length it delivers is reported beside the
# number of episodes it rests on, because at a wide limit that count is a
# handful of excursions rather than a rate.
EWMA_L_CANDIDATES = np.arange(2.0, 15.01, 0.25)

# CUSUM slack and decision interval, in standard deviations.
CUSUM_K = 0.5
CUSUM_H = 5.0

# Coincidence window for the joint alarm.
JOINT_WINDOW = '6h'

# No event window is parameterised. The summer-2026 stretch is kept out of the
# reference window above, but it is not a test and no result is stated from it
# (D12).

# The false-alarm budget the charts are tuned to, in days between false alarms
# on the in-control reference stretch. Every detection figure in this study is
# only comparable at a stated run length, and this is it.
TARGET_ARL_DAYS = 90.0

# ---------------------------------------------------------------------------
# Detectability sweep
# ---------------------------------------------------------------------------
# Injected departures, in millidegrees and hours. The magnitudes bracket the
# residual's own scale so that the curve crosses from undetectable to certain
# inside the swept range; the durations span a working day to a fortnight.
DETECT_MAGNITUDES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DETECT_DURATIONS = ('6h', '24h', '72h', '168h', '336h')

# The three mechanisms swept, each a shape a three-leaf wall can produce:
#   'amplitude' - the daily swing grows while its timing and mean hold, which is
#                 what loss of composite action between the leaves looks like;
#   'phase'     - the response arrives earlier or later against the same
#                 forcing, which is a change in the thermal path rather than in
#                 stiffness, such as water in the core;
#   'drift'     - a slow monotone accumulation, the shape of mortar creep,
#                 thermal ratcheting or settlement.
DETECT_KINDS = ('amplitude', 'phase', 'drift')

# Timing shifts probed for the phase mechanism, in hours. Each is converted into
# the residual amplitude it implies through the fitted daily amplitude, so the
# result is quoted as the shift an engineer would picture rather than as a
# millidegree figure with no mechanism attached.
DETECT_PHASE_SHIFTS_H = (0.25, 0.5, 1.0, 2.0)

# Drift rates probed, in millidegrees per year.
DETECT_DRIFT_RATES = (1.0, 2.0, 5.0, 10.0, 20.0)

# How long after a departure ends an alarm still counts as having found it.
DETECT_RESPONSE_WINDOW = '24h'

# ---------------------------------------------------------------------------
# Model B - forecast
# ---------------------------------------------------------------------------
# Hourly, because the twenty-minute first difference is noise-dominated:
# measured on this record its lag-one autocorrelation is -0.058, the signature
# of additive noise on the level, while the hourly difference retains +0.349.
# Step 3 exports the comparison as NP_04.
MODEL_B_TARGET = 'change'

# Autoregressive window, chosen from NP_03 rather than by habit: it is the
# longest window that leaves enough training windows on this record's
# contiguity. Read the table before changing it.
MODEL_B_LAGS = 24

# Forecast length and the horizons kept from it.
MODEL_B_FORECASTS = 48
MODEL_B_HORIZONS = (1, 3, 6, 12, 24, 48)

# Predictor history. Past values only: a forecast that consumed a future
# observed air temperature would be answering a different question.
MODEL_B_REGRESSOR_LAGS = 12

# The ablation ladder. Each rung adds one thing, so the increment it buys is
# attributable. The battery control is not a silent channel - study 03 measured
# r = -0.673 for it in the diurnal band - so it marks a conservative floor
# rather than a zero, and a driver is credited only when it clears that floor.
MODEL_B_SPECIFICATIONS = {
    'AR only': (),
    'AR + tair': ('tair',),
    'AR + tair + rh': ('tair', 'rh'),
    'AR + batt (control)': ('batt',),
}

MODEL_B_EPOCHS = 30
MODEL_B_QUANTILES = (0.05, 0.95)
MODEL_B_SEED = 0
MODEL_B_INITIAL_SEGMENTS = 20
MODEL_B_FOLDS = 5
MODEL_B_MIN_SEGMENT = 96
MODEL_B_REFIT_EACH_FOLD = False

# Block bootstrap for the paired skill comparison. Residuals are autocorrelated,
# so an unpaired comparison would overstate significance; the block length is
# adapted to the horizon inside paired_mae_skill.
BOOTSTRAP_BLOCK_HOURS = 24
BOOTSTRAP_REPETITIONS = 2000

# ---------------------------------------------------------------------------
# Gap closure
# ---------------------------------------------------------------------------
# The reconstruction is accepted only if the median absolute closure error
# across bracketed gaps stays below this many millidegrees. It is set against
# the minimum detectable step measured in step 7b: a reconstruction whose error
# exceeds what the monitor can detect would manufacture alarms.
GAP_CLOSURE_TOLERANCE_MDEG = 1.0

# %% [markdown]
# ## 1 · The window and the on-structure record
#
# Both the response and the other channels of the package come from Study 01's
# verdict-aware loaders: accepted values are kept and everything rejected or
# absent stays missing. Radiation keeps its accepted values only, and wall
# temperature uses the filtered probe channel Study 01 exported. Nothing is
# interpolated here, and nothing is interpolated before drawing.

# %%
inclination, inclination_provenance = proxies.load_response(
    ARCHIVE_CSV, column=TARGET_COLUMN, honour_spike=True,
    freq=NATIVE_FREQ, tz=site.SITE_TZ, min_count=1)

sensor_current, sensor_provenance = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_CURRENT, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor_legacy, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_LEGACY, freq=NATIVE_FREQ,
    tz=site.SITE_TZ, honour_suspect=True, min_count=1)
sensor = proxies.join_eras([sensor_current, sensor_legacy])

record = proxies.harmonise([sensor, inclination.to_frame()], freq=NATIVE_FREQ)
window = record.loc[site.to_utc(pd.DatetimeIndex([SEGMENT_START]))[0]:]

print(f'Window: {window.index.min()} to {window.index.max()} '
      f'({len(window):,} slots on the {NATIVE_FREQ} grid)')
print(f'First accepted inclination inside the window: '
      f'{window[TARGET_COLUMN].dropna().index.min()}')

# Coverage per channel over the window, which is what the report quotes. It
# counts accepted values against the slots of the window, and says nothing yet
# about how the missing time is distributed — that is the next movement.
coverage = pd.DataFrame({
    'accepted': window[list(ON_STRUCTURE_COLUMNS)].notna().sum(),
    'coverage': window[list(ON_STRUCTURE_COLUMNS)].notna().mean(),
})
coverage.index.name = 'channel'
display(coverage)

coverage.reset_index().to_csv(OUTPUT_DIR / 'NP_01_window_coverage.csv',
                              index=False)

# %%
figures.plot_channel_panels(
    window, list(ON_STRUCTURE_COLUMNS),
    title='The on-structure record after the 2022-2023 outage',
    save_path=str(OUTPUT_DIR), filename='NP_F01_on_structure_record')
plt.show()

# %% [markdown]
# ## 2 · The anatomy of what is missing
#
# Study 01 measured how much of the record is absent and recorded explicitly
# that it had not measured how that absence is *shaped*. It is measured here,
# because the shape decides what kind of problem filling it is: recovering a
# scattered sample from its immediate neighbours is a different proposition
# from reconstructing a season. The same table also decides how long an
# autoregressive window this record can afford, since a window can only be
# trained inside a run of complete rows long enough to hold it.
#
# ### Parameter Tuning Guidance
#
# **`SURVIVAL_LAG_HOURS`** — autoregressive window lengths to evaluate, in
# hours. Accepts any increasing sequence of positive integers; default
# `(4, 8, 12, 24, 48)`. Raising the largest value costs training windows
# quadratically on a fragmented record: each configuration is reported so the
# choice made in step 6 can be read off the table rather than assumed.
#
# **`SURVIVAL_FORECAST_HOURS`** — forecast lengths to pair with each window;
# default `(8, 24)`. A configuration survives only in segments at least
# `lag + forecast` long, so this parameter and the one above trade against each
# other and are reported jointly.

# %%
gaps = prediction.gap_inventory(window[TARGET_COLUMN], freq=NATIVE_FREQ)
print(f'{len(gaps):,} gaps, {gaps["duration_h"].sum():,.0f} missing hours')
display(gaps.groupby('gap_class')
        .agg(gaps=('n_slots', 'size'), hours=('duration_h', 'sum')))

survival = prediction.segment_survival(
    window.rename(columns={TARGET_COLUMN: 'y'}),
    required=['y'] + [f'{name}_str' for name in PREDICTOR_COLUMNS],
    lag_hours=list(SURVIVAL_LAG_HOURS),
    forecast_hours=list(SURVIVAL_FORECAST_HOURS),
    freq=NATIVE_FREQ)
display(survival)

gaps.to_csv(OUTPUT_DIR / 'NP_02_gap_inventory.csv', index=False)
survival.to_csv(OUTPUT_DIR / 'NP_03_segment_survival.csv', index=False)

figures.plot_gap_anatomy(
    gaps, title='How the missing time is shaped',
    save_path=str(OUTPUT_DIR), filename='NP_F02_gap_anatomy')
figures.plot_segment_survival(
    survival, title='Training windows surviving contiguous segmentation',
    save_path=str(OUTPUT_DIR), filename='NP_F03_segment_survival')
plt.show()

# %% [markdown]
# ## 3 · Which cadence, and which target
#
# Two choices are settled here by measurement rather than by preference: whether
# the level or its first difference is the quantity to score, and at which
# spacing. A level whose lag-one autocorrelation approaches unity cannot be
# scored honestly, because any error metric computed against it measures the
# sampling interval rather than the model. A first difference whose lag-one
# autocorrelation is *negative* is dominated by measurement noise on the level
# rather than by the increment it is meant to carry. Both quantities are
# reported at both candidate cadences, together with the coupling to air
# temperature and the drift each implies.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_FREQ_A`** — grid for the decomposition and expectation model;
# default `'20min'`, the archive's native spacing and the rate at which a
# deployed system receives a reading. The anomaly question does not require
# differencing, so the noise that spoils the twenty-minute difference does not
# affect it.
#
# **`MODEL_FREQ_B`** — grid for the forecast model; default `'1h'`. Set it to
# `'20min'` only if `NP_04` shows a non-negative change autocorrelation there;
# on this record it does not.

# %%
# No era label is passed (D9). Study 01 anchored the two instrument eras once,
# as a single series, and this study reads that product as its raw material;
# labelling the changeover here would only invite a second anchoring.
cadence = prediction.cadence_evidence(
    window[TARGET_COLUMN], window['tair_str'], era=None,
    cadences=(MODEL_FREQ_A, MODEL_FREQ_B), freq=NATIVE_FREQ)
display(cadence)

cadence.to_csv(OUTPUT_DIR / 'NP_04_cadence_evidence.csv', index=False)
figures.plot_cadence_evidence(
    cadence, title='What fixes the cadence and the target',
    save_path=str(OUTPUT_DIR), filename='NP_F04_cadence_evidence')
plt.show()

# %% [markdown]
# ## 4 · What the record is made of
#
# The level is decomposed into a trend, a daily cycle, the response to the two
# environmental channels measured beside it, and a remainder. The level is used
# here and nowhere else in this study: its lag-one autocorrelation is 0.998, so
# an error metric computed against it would measure the sampling interval rather
# than the model. What it is good for is the trend, which no differenced target
# can recover — the mean of the gap-safe change implies −65.7 mdeg/yr at twenty
# minutes and +0.97 mdeg/yr at one hour on this same record, because segment
# endpoints do not sample the diurnal cycle uniformly.
#
# Nothing is interpolated. Rows missing any required channel are dropped, the
# remainder is split into contiguous segments, and each segment is handed to
# NeuralProphet under its own identifier, so no fitted window ever spans a gap.
# Imputation is disabled explicitly: with `impute_missing=True`, its default,
# NeuralProphet silently fabricates gaps of at least thirty hours.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_A_LAGS`** — autoregressive lags; must stay `0`. Any positive value
# transfers the diurnal cycle from the seasonal component into the
# autoregressive one and makes the decomposition unreadable as physics.
#
# **`MODEL_A_GROWTH`** — trend of the attribution fit; `'linear'` or `'off'`,
# default `'linear'`. Under `'off'` the trend is a constant and the drift
# disappears, which is why the attribution fit keeps it.
#
# **`MODEL_A_MONITOR_GROWTH`** — trend of the monitoring fit; default `'off'`.
# This is the fit whose residual becomes the expectation error of step 6 and the
# control statistic of step 7. It carries no trend because an extrapolated one
# is the largest error in the residual by a factor of three, and because the
# drift belongs in front of the detector rather than behind it.
#
# **`MODEL_A_CHANGEPOINTS`** — number of trend changepoints, placed on covered
# time; default `12`, roughly one per quarter of the window. More changepoints
# track shorter movements at the cost of absorbing signal that belongs to the
# seasonal or regressor terms.
#
# **`MODEL_A_YEARLY_CANDIDATES`** — which annual settings are fitted for the
# comparison; both, so the cost of the annual term is measured and reported
# rather than assumed.
#
# **`MODEL_A_YEARLY`** — which of them is carried forward; default `True`. This
# is deliberately not the held-out winner. Model A is read as an attribution of
# variation to named causes, and an annual cycle left out of the model does not
# cease to exist: it moves into the residual, where the control charts of step 7
# would read it as a structural departure. Set it to `False` only for a study
# whose purpose is forecast accuracy rather than attribution.
#
# **`MODEL_A_TRAIN_END`** — the frozen training origin. Everything after it is
# out of sample. Moving it later buys training data and costs evaluation data.
#
# **`MODEL_A_MIN_SEGMENT`** — shortest usable run, in slots; default `72`, one
# day at twenty minutes.

# %%
frame_a = (window.rename(columns={TARGET_COLUMN: 'y'})
           .rename(columns={f'{name}_str': name for name in PREDICTOR_COLUMNS})
           .loc[:, ['y'] + list(PREDICTOR_COLUMNS)])

segmented_a = prediction.contiguous_segments(
    frame_a, required=['y'] + list(PREDICTOR_COLUMNS),
    min_length=MODEL_A_MIN_SEGMENT, freq=MODEL_FREQ_A)

train_a = segmented_a.loc[:MODEL_A_TRAIN_END]
test_a = segmented_a.loc[MODEL_A_TRAIN_END:]
changepoints_a = prediction.covered_changepoints(
    train_a.index, MODEL_A_CHANGEPOINTS)

print(f'Model A: {len(train_a):,} training rows in '
      f'{train_a["segment_id"].nunique()} segments, '
      f'{len(test_a):,} evaluation rows')

# %%
fits = {}
for yearly in MODEL_A_YEARLY_CANDIDATES:
    model, predictions = prediction.neuralprophet_backtest(
        train_a, test_a, regressors=PREDICTOR_COLUMNS, task='nowcast',
        n_lags=MODEL_A_LAGS, epochs=MODEL_A_EPOCHS, yearly=yearly,
        quantiles=MODEL_A_QUANTILES, seed=MODEL_A_SEED,
        growth=MODEL_A_GROWTH, changepoints=changepoints_a,
        freq=MODEL_FREQ_A)
    fits[yearly] = (model, predictions)
    scores = prediction.score_predictions(predictions, [])
    print(f'yearly={yearly}: out-of-sample MAE {scores["mae"].iloc[0]:.3f} mdeg')

# The held-out winner is reported, and then not obeyed: see MODEL_A_YEARLY.
mae_optimal = min(
    fits, key=lambda flag: prediction.score_predictions(
        fits[flag][1], [])['mae'].iloc[0])
model_a, predictions_a = fits[MODEL_A_YEARLY]
print(f'Lowest held-out MAE at yearly={mae_optimal}; '
      f'carried forward yearly={MODEL_A_YEARLY} '
      f'({"agrees" if mae_optimal == MODEL_A_YEARLY else "overridden for attribution"})')

# %%
# The monitoring fit: same data, same regressors, same annual term, no trend.
model_m, predictions_m = prediction.neuralprophet_backtest(
    train_a, test_a, regressors=PREDICTOR_COLUMNS, task='nowcast',
    n_lags=MODEL_A_LAGS, epochs=MODEL_A_EPOCHS, yearly=MODEL_A_YEARLY,
    quantiles=MODEL_A_QUANTILES, seed=MODEL_A_SEED,
    growth=MODEL_A_MONITOR_GROWTH, changepoints=None, n_changepoints=0,
    freq=MODEL_FREQ_A)
monitor_scores = prediction.score_predictions(predictions_m, [])
print(f'Monitoring fit (growth={MODEL_A_MONITOR_GROWTH!r}): '
      f'out-of-sample MAE {monitor_scores["mae"].iloc[0]:.3f} mdeg')

# %% [markdown]
# ## 5 · The components, and whether they agree with Study 03
#
# Two fits are carried from here, and they answer different questions. The
# **attribution fit** keeps a piecewise-linear trend and says what the record is
# made of; its trend is the study's drift estimate, and it is read over the
# training window, where a fitted trend is constrained by observations. The
# **monitoring fit** is the same model without that trend, and its residual is
# what step 6 judges a new reading against and step 7 charts.
#
# The split is forced by the record rather than chosen for elegance. Measured
# over the evaluation stretch, the attribution fit's residual has a standard
# deviation of 41.9 mdeg and reaches -194, against 13.8 and -43 for the same
# model with no trend: past its last changepoint the trend is an extrapolation,
# and on a record with a 103-day hole it is by far the largest error in the
# residual. Monitoring on it would mean alarming on the model's own
# extrapolation. Dropping the trend leaves the drift in the residual, which is
# where a monitoring system should meet it.
#
# The fitted air-temperature contribution is the one number in this study that
# can be checked against an independent measurement. Study 03 screened the same
# response against the same channel by a completely different method — a lag and
# gain scan on the diurnal band — and measured **−2.79 mdeg/°C** with
# `r = −0.957`, a value that survived substitution of the ground station
# (−2.23) and ERA5 (−2.04) for the on-structure sensor. If this decomposition
# reproduces it, two unrelated methods agree on a physical constant. If it does
# not, that disagreement is the study's finding and the work stops here rather
# than proceeding to build an anomaly detector on a model that does not describe
# the wall.
#
# The gain is fitted on the compensated channel, which this study takes as its
# raw data. Whether Study 01's compensation is correctly sized is Study 01's
# question, and it is not reopened here (D10): the number below is what the wall
# does after that correction, which is the only quantity a monitoring system
# ever sees.

# %%
# The attribution fit says what the record is made of.
components_a = prediction.decompose_components(
    model_a, segmented_a, regressors=PREDICTOR_COLUMNS)

# The monitoring fit, whose residual step 6 refits on a schedule and then
# charts. Keeping the two apart is the whole point of fitting twice.
components_m = prediction.decompose_components(
    model_m, segmented_a, regressors=PREDICTOR_COLUMNS)

shares = prediction.component_variance_shares(components_a)
diagnostics = pd.concat([
    prediction.residual_diagnostics(
        components_a['residual'], lags=RESIDUAL_LAGS).assign(fit='attribution'),
    prediction.residual_diagnostics(
        components_m['residual'], lags=RESIDUAL_LAGS).assign(
            fit='monitoring, frozen'),
], ignore_index=True)
diagnostics = diagnostics[['fit'] + [c for c in diagnostics.columns
                                     if c != 'fit']]
display(shares)
display(diagnostics)

shares.to_csv(OUTPUT_DIR / 'NP_05_component_shares.csv', index=False)
diagnostics.to_csv(OUTPUT_DIR / 'NP_08_residual_diagnostics.csv', index=False)

# Drawn over the whole record, not merely the evaluation stretch. The annual
# term completes about three cycles here, and on the post-training window alone
# the 103-day outage falls exactly across a crest, which makes a fitted annual
# component look like a monotone ramp.
figures.plot_decomposition_stack(
    components_a, freq=MODEL_FREQ_A,
    title='What the inclination record is made of',
    save_path=str(OUTPUT_DIR), filename='NP_F05_decomposition_stack')
plt.show()

# %%
# The learned thermal gain, set against Study 03's three independent statements
# of it.
paired = pd.concat([components_a['future_regressor_tair'],
                    segmented_a['tair']], axis=1).dropna()
paired.columns = ['contribution', 'tair']
learned_gain = np.polyfit(paired['tair'], paired['contribution'], 1)[0]

gains = pd.DataFrame([
    {'source': 'Model A, compensated channel', 'gain_mdeg_per_degC': learned_gain,
     'method': 'NeuralProphet future regressor', 'n': len(paired)},
    {'source': 'Study 03, diurnal band', 'gain_mdeg_per_degC': -2.79,
     'method': 'lag and gain scan', 'n': np.nan},
    {'source': 'Study 03, ground station', 'gain_mdeg_per_degC': -2.23,
     'method': 'lag and gain scan', 'n': np.nan},
    {'source': 'Study 03, ERA5', 'gain_mdeg_per_degC': -2.04,
     'method': 'lag and gain scan', 'n': np.nan},
])
display(gains)
gains.to_csv(OUTPUT_DIR / 'NP_06_learned_gains.csv', index=False)

# %%
# The drift, read from the attribution fit's trend over the training window
# only. Beyond the last changepoint the trend is an extrapolation and says
# nothing about the wall.
trend_in_sample = components_a['trend'].loc[:MODEL_A_TRAIN_END].dropna()
elapsed_days = ((trend_in_sample.index - trend_in_sample.index[0])
                / pd.Timedelta(days=1)).to_numpy()
drift_per_year = float(
    np.polyfit(elapsed_days, trend_in_sample.to_numpy(), 1)[0] * 365.0)
print(f'Fitted drift over the training window: {drift_per_year:+.2f} mdeg/yr '
      f'({trend_in_sample.index.min().date()} to '
      f'{trend_in_sample.index.max().date()})')

# %% [markdown]
# ## 6 · Is this reading the one that was expected?
#
# The model is asked a conditional question, not a predictive one: given the air
# temperature and relative humidity recorded in the same acquisition record as
# this inclination, what inclination did the wall owe? Consuming those two
# channels at the same instant is therefore not leakage — they arrive together,
# twenty minutes at a time, and a monitoring system has them in hand at the
# moment it must judge. The forecast model in step 8 is held to the opposite
# rule and may consume only past predictor values; the two are never scored on
# the same table.
#
# The expectation is the monitoring fit's, not the attribution fit's, and it is
# refitted on a schedule rather than frozen. Two measurements force both
# choices. Past its last changepoint the attribution fit's trend is an
# extrapolation, so an operator asking "is this reading normal" would be
# answered mostly by how far that extrapolation has wandered. And a fit frozen
# at the training origin goes stale: applied to the following year it sits
# 28.75 mdeg away from the record, with 82% of its mean square error in that
# single offset rather than in scatter. Refitting every thirty days holds the
# expectation to the drift a month can accumulate — about 0.23 mdeg at the
# measured rate — which is what a deployed system does and what this study must
# therefore report.
#
# The interval is conformal rather than the model's own. NeuralProphet's
# quantile regression fits the residual's core, and this residual has a tight
# core with fat tails: its nominal 90% interval covers 68.7% even in-sample.
# Empirical residual quantiles, calibrated on rows the interval is never scored
# against, make coverage mean what it claims.
#
# An interval that is drawn but never checked is decoration, so the 90 %
# interval is scored on three counts: how often it actually contains the
# observation, how wide it had to be to manage that, and the pinball loss of
# each quantile on its own terms. Coverage alone can always be bought by
# widening.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_A_QUANTILES`** — the interval's quantiles; default `(0.05, 0.95)`,
# a nominal 90 % interval. Narrower quantiles alarm more often at a fixed
# threshold and shorten detection delay at the cost of false alarms.
#
# **`NOWCAST_INTERVAL_ALPHA`** — nominal miss rate used by the Winkler interval
# score; default `0.10`, matching the quantiles above. Change both together or
# the score is no longer comparable with the coverage beside it.
#
# **`NOWCAST_VIEW`, `COMPONENT_VIEW`** — the stretches drawn in `NP_F07` and
# `NP_F06`. They change what a reader sees and nothing that is computed.
#
# **`ROLLING_REFIT_EVERY`** — how often the monitoring model is refitted;
# default `'30d'`. Shorter keeps the expectation fresher and costs one fit per
# window; longer lets the same staleness back in. This is the operational
# parameter a deployment actually sets, and it is reported beside every number
# derived from the rolling residual.
#
# **`ROLLING_MIN_TRAIN`** — history required before the first window is scored;
# default `'365d'`, one full annual cycle. It also decides where the usable
# residual record begins, so raising it buys better early fits and costs months
# of monitored record. Below a year the annual term is fitted to less than one
# cycle and is not identifiable; the residuals of those windows are the widest
# in the record and they contaminate whatever the conformal interval is
# calibrated on.
#
# **`CONFORMAL_CALIBRATION_END`** — the last instant whose residuals may inform
# the interval; default `'2025-06-01'`. Rows after it are judged, never used to
# calibrate what judges them.

# %%
# The naive scale for MASE is the in-sample mean absolute one-step change. It is
# computed on the training rows only: a scale taken from the evaluation rows
# would make the metric self-referential.
naive_scale_a = float(
    prediction.hourly_change(train_a['y'], freq=MODEL_FREQ_A).abs().mean())

# The rolling expectation: one fit per window, never more than
# ROLLING_REFIT_EVERY old, over the whole record rather than the held-out tail.
rolling_raw = prediction.rolling_nowcast(
    segmented_a, regressors=PREDICTOR_COLUMNS,
    refit_every=ROLLING_REFIT_EVERY, min_train=ROLLING_MIN_TRAIN,
    freq=MODEL_FREQ_A, n_lags=MODEL_A_LAGS, epochs=MODEL_A_EPOCHS,
    yearly=MODEL_A_YEARLY, quantiles=MODEL_A_QUANTILES, seed=MODEL_A_SEED,
    growth=MODEL_A_MONITOR_GROWTH, changepoints=None, n_changepoints=0)

rolling = prediction.conformal_interval(
    rolling_raw, alpha=NOWCAST_INTERVAL_ALPHA,
    calibration_end=CONFORMAL_CALIBRATION_END)

print(f'Rolling expectation: {len(rolling):,} rows from '
      f'{rolling["origin"].nunique()} refits, '
      f'{rolling["ds"].min().date()} to {rolling["ds"].max().date()}')

nowcast_scores = pd.concat([
    prediction.score_predictions(
        rolling, [], naive_scale=naive_scale_a,
        alpha=NOWCAST_INTERVAL_ALPHA).assign(fit='rolling, conformal'),
    prediction.score_predictions(
        predictions_m, [], naive_scale=naive_scale_a,
        alpha=NOWCAST_INTERVAL_ALPHA).assign(fit='frozen monitoring'),
    prediction.score_predictions(
        predictions_a, [], naive_scale=naive_scale_a,
        alpha=NOWCAST_INTERVAL_ALPHA).assign(fit='frozen attribution'),
], ignore_index=True)
nowcast_scores.insert(0, 'model', 'Model A · tair + rh')
nowcast_scores.insert(1, 'fit', nowcast_scores.pop('fit'))
display(nowcast_scores)
nowcast_scores.to_csv(OUTPUT_DIR / 'NP_07_nowcast_metrics.csv', index=False)

reported = nowcast_scores.iloc[0]
print(f'Rolling, conformal — nominal coverage '
      f'{1 - NOWCAST_INTERVAL_ALPHA:.0%}, observed '
      f'{reported["coverage_q05_q95"]:.1%}, median width '
      f'{reported["width_q05_q95"]:.2f} mdeg, MAE '
      f'{reported["mae"]:.2f} mdeg, bias {reported["bias"]:+.2f}, '
      f'MASE {reported["mase"]:.2f}')

# This residual, not the frozen fit's, is what step 7 charts and what every
# detectability number is quoted against.
residual_a = (rolling.set_index('ds')['y']
              - rolling.set_index('ds')['yhat']).dropna().sort_index()
print(f'Monitoring residual: {len(residual_a):,} slots, '
      f'sd {residual_a.std():.2f} mdeg, '
      f'MAD {float((residual_a - residual_a.median()).abs().median()):.2f}')

# NP_08 is rewritten here rather than in step 5, so that it describes the
# residual the study actually charts alongside the two frozen fits it is being
# preferred to.
diagnostics = pd.concat([
    prediction.residual_diagnostics(
        residual_a, lags=RESIDUAL_LAGS).assign(fit='rolling, conformal'),
    diagnostics,
], ignore_index=True)
diagnostics = diagnostics[['fit'] + [c for c in diagnostics.columns
                                     if c != 'fit']]
display(diagnostics)
diagnostics.to_csv(OUTPUT_DIR / 'NP_08_residual_diagnostics.csv', index=False)

# %%
# One month of the evaluation period, drawn at readable density.
view = rolling.set_index('ds').sort_index().loc[
    NOWCAST_VIEW[0]:NOWCAST_VIEW[1]]
figures.plot_prediction_band(
    view['y'], view['yhat'], view['q05'], view['q95'], freq=MODEL_FREQ_A,
    title='Observed inclination against what the measured environment predicted',
    save_path=str(OUTPUT_DIR), filename='NP_F07_observed_vs_expected')

# The two interpretable components on their own axes: the daily cycle the model
# fitted, and the response it attributes to the measured environment. These come
# from the attribution fit, which is the one that claims to say what the record
# is made of.
figures.plot_decomposition_stack(
    components_a.loc[COMPONENT_VIEW[0]:COMPONENT_VIEW[1]],
    columns=['season_daily', 'future_regressor_tair', 'future_regressor_rh'],
    freq=MODEL_FREQ_A,
    title='The daily cycle and the response to the measured environment',
    save_path=str(OUTPUT_DIR), filename='NP_F06_daily_cycle_and_response')
plt.show()

# %% [markdown]
# ### 6b · Do the components survive being re-estimated?
#
# A component estimated once is an assertion; one that survives being
# re-estimated on disjoint stretches of the record is a finding. The attribution
# model is refitted on each chronological third of the training period and its
# two interpretable quantities — the trend slope and the air-temperature gain —
# are compared across them. A gain that changes sign between thirds, or a slope
# that varies by more than its own magnitude, is reported as instability rather
# than averaged away.

# %%
stability_rows = []
thirds = np.array_split(train_a.index.unique(), 3)
for number, block in enumerate(thirds, 1):
    part = train_a.loc[block[0]:block[-1]]
    if part['segment_id'].nunique() < 2:
        continue
    model_part, _ = prediction.neuralprophet_backtest(
        part, part, regressors=PREDICTOR_COLUMNS, task='nowcast',
        n_lags=MODEL_A_LAGS, epochs=MODEL_A_EPOCHS, yearly=MODEL_A_YEARLY,
        quantiles=(), seed=MODEL_A_SEED, growth=MODEL_A_GROWTH,
        changepoints=prediction.covered_changepoints(
            part.index, max(2, MODEL_A_CHANGEPOINTS // 3)),
        freq=MODEL_FREQ_A)
    parts = prediction.decompose_components(
        model_part, part, regressors=PREDICTOR_COLUMNS)
    paired_part = pd.concat(
        [parts['future_regressor_tair'], part['tair']], axis=1).dropna()
    paired_part.columns = ['contribution', 'tair']
    elapsed_days = ((parts.index - parts.index[0])
                    / pd.Timedelta(days=1)).to_numpy()
    stability_rows.append({
        'block': f'third {number}',
        'start': part.index.min(),
        'end': part.index.max(),
        'tair_gain_mdeg_per_degC': float(
            np.polyfit(paired_part['tair'], paired_part['contribution'], 1)[0]),
        'trend_mdeg_per_year': float(
            np.polyfit(elapsed_days, parts['trend'].to_numpy(), 1)[0] * 365.0),
    })

stability = pd.DataFrame(stability_rows)
display(stability)
stability.to_csv(OUTPUT_DIR / 'NP_16_component_stability.csv', index=False)

signs = np.sign(stability['tair_gain_mdeg_per_degC'])
print('Thermal gain keeps its sign across thirds:', bool(signs.nunique() == 1))

# %% [markdown]
# ## 7 · Judging a departure
#
# The residual of step 6 is what remains after the daily cycle and the measured
# environment have been accounted for. A departure is a stretch where that
# remainder stops behaving as it did over the reference window.
#
# Two charts run together because they fail in different ways. The exponentially
# weighted average answers "has the level moved and stayed moved", which is the
# shape a structural departure takes; the cumulative sum accumulates evidence
# and finds a shift too small to breach a limit on any single sample, provided
# it persists. An alarm is raised only where both agree within a coincidence
# window, which trades a little sensitivity for a large reduction in isolated
# false alarms.
#
# A detector can be made to look perfect by never alarming, so the settings are
# fixed the other way round: the limit width is chosen to meet a stated
# false-alarm budget on the in-control stretch, and every detection figure that
# follows is quoted at that budget.
#
# ### Parameter Tuning Guidance
#
# **`REFERENCE_START`, `REFERENCE_END`** — the in-control window, and also where
# the monitored record begins. Must exclude any stretch suspected of carrying a
# departure; on this record it ends before the one Study 01 flagged in summer
# 2026. That exclusion is a precaution about calibration, not a claim that the
# detector finds it. The start excludes the rolling expectation's burn-in, whose
# size is measured in the parameter cell's comment; moving it earlier admits
# months in which the expectation is wrong by tens of millidegrees for a reason
# that has nothing to do with the wall.
#
# **`EWMA_L_CANDIDATES`** — the limit widths swept. The range must bracket the
# width that meets the budget, and on a residual this autocorrelated that width
# is far above the textbook range. Widening the range is not the same as
# widening the limit until the alarms stop: the budget is fixed first, and the
# width is whatever meets it.
#
# **`TARGET_ARL_DAYS`** — days of watched time per false alarm; default `90`.
# Lower it and the system finds smaller movements sooner while crying wolf more
# often. This is the operator's dial, and it is the axis every other detection
# number is quoted against.
#
# **`EWMA_LAMBDA`** — smoothing weight; default `0.05`, tuned for sustained
# shifts rather than spikes. **`EWMA_L`** is swept to meet the budget rather
# than set by hand.
#
# **`CUSUM_K`, `CUSUM_H`** — slack and decision interval, in standard
# deviations; defaults `0.5` and `5.0`, the textbook pair for detecting a
# one-sigma shift quickly.
#
# **`JOINT_WINDOW`** — how close in time the two charts must agree; default
# `'6h'`. Wider admits more coincidences and raises the false-alarm rate.

# %%
# The charts need a regular grid. `alarm_episodes` and `average_run_length`
# derive every duration they report from a slot count times one spacing, and
# refuse an index whose rows were dropped across an outage, because an episode
# would otherwise be reported as spanning time the record does not cover. The
# rolling residual exists only where a window predicted, so it is placed back
# onto the complete twenty-minute grid with the missing slots carried as NaN.
# The charts skip a NaN rather than reading it as a zero departure: the
# statistic is held across a gap rather than relaxed towards the centre, which
# is what a monitoring system does when readings stop arriving.
residual_grid = residual_a.reindex(
    pd.date_range(residual_a.index.min(), residual_a.index.max(),
                  freq=MODEL_FREQ_A)).loc[REFERENCE_START:]
print(f'Residual on the grid: {len(residual_grid):,} slots from '
      f'{residual_grid.index.min().date()}, '
      f'{residual_grid.notna().sum():,} observed '
      f'({residual_grid.notna().mean():.1%})')
print(f'Residual autocorrelation: {residual_grid.autocorr(1):.4f} at one slot, '
      f'{residual_grid.autocorr(72):.4f} at twenty-four hours. A control chart '
      f'assumes neither.')

reference_a = monitoring.reference_stats(
    residual_grid, start=REFERENCE_START, end=REFERENCE_END, robust=True)
in_control = residual_grid.loc[REFERENCE_START:REFERENCE_END]
print(f'Reference: mu={reference_a["mu"]:.3f} mdeg, '
      f'sigma={reference_a["sigma"]:.3f} mdeg, n={reference_a["n"]:,}, '
      f'sd {in_control.std():.3f} mdeg, window '
      f'{in_control.dropna().index.min()} to '
      f'{in_control.dropna().index.max()}')

# Choose the limit width that meets the false-alarm budget on the in-control
# stretch, rather than accepting a conventional value and reporting whatever
# rate it happens to give.
budget = []
for candidate_L in EWMA_L_CANDIDATES:
    ewma = monitoring.ewma_chart(in_control, reference_a['mu'],
                                 reference_a['sigma'],
                                 lam=EWMA_LAMBDA, L=candidate_L)
    cusum = monitoring.cusum_chart(in_control, reference_a['mu'],
                                   reference_a['sigma'],
                                   k=CUSUM_K, h=CUSUM_H)
    joint = monitoring.joint_alarm(ewma['alarm'], cusum['alarm'],
                                   window=JOINT_WINDOW)
    arl = monitoring.average_run_length(joint, freq=MODEL_FREQ_A)
    budget.append({'L': candidate_L, **arl})

budget = pd.DataFrame(budget)
display(budget)

# No candidate meeting the budget means the residual is not in control over the
# reference window. That is a statement about the expectation, not a reason to
# widen the limit until the alarms stop, so it fails loudly here rather than
# carrying a missing limit into every number below.
meeting = budget.loc[budget['arl_days'] >= TARGET_ARL_DAYS, 'L']
if meeting.empty:
    raise RuntimeError(
        f'No limit width between {budget["L"].min()} and {budget["L"].max()} '
        f'reaches {TARGET_ARL_DAYS:.0f} days per false alarm; the best is '
        f'{budget["arl_days"].max():.1f} days. The reference residual is not '
        f'in control - revisit the expectation rather than the limit.')
EWMA_L = float(meeting.min())
chosen = budget.loc[budget['L'] == EWMA_L].iloc[0]
print(f'Chosen L = {EWMA_L}, delivering {chosen["arl_days"]:.1f} days per '
      f'false alarm against a {TARGET_ARL_DAYS:.0f}-day budget, on '
      f'{int(chosen["n_episodes"])} episodes over '
      f'{chosen["hours"] / 24.0:.0f} watched days.')
# The run length rests on that episode count, and at a wide limit it is a
# handful of excursions rather than a rate, so it is quoted as an order of
# magnitude and never as a precise figure.
if chosen['n_episodes'] < 5:
    print(f'  Caution: {int(chosen["n_episodes"])} episodes is too few for the '
          f'run length to be a precise estimate. It bounds the false-alarm '
          f'rate rather than measuring it.')

# %%
ewma_a = monitoring.ewma_chart(residual_grid, reference_a['mu'],
                               reference_a['sigma'],
                               lam=EWMA_LAMBDA, L=EWMA_L)
cusum_a = monitoring.cusum_chart(residual_grid, reference_a['mu'],
                                 reference_a['sigma'], k=CUSUM_K, h=CUSUM_H)
alarm_a = monitoring.joint_alarm(ewma_a['alarm'], cusum_a['alarm'],
                                 window=JOINT_WINDOW)
episodes_a = monitoring.alarm_episodes(alarm_a, ewma_a['z'])
display(episodes_a)
print(f'{len(episodes_a)} alarming episodes over the whole monitored record, '
      f'{float(episodes_a["duration_h"].sum()) / 24.0:.1f} days in alarm of '
      f'{len(residual_grid) * 20 / 60 / 24:.0f} watched')

episodes_a.to_csv(OUTPUT_DIR / 'NP_09_alarm_episodes.csv', index=False)
figures.plot_control_chart(
    ewma_a, statistic='ewma', episodes=episodes_a, freq=MODEL_FREQ_A,
    title='Residual control chart, with alarming episodes shaded',
    save_path=str(OUTPUT_DIR), filename='NP_F08_control_chart')
plt.show()

# %% [markdown]
# ### 7b · Which damage signatures would be found, and how late
#
# One large event cannot state a detector's sensitivity. Departures of known
# size, length and *shape* are injected into the residual instead, the charts
# are re-run with the reference statistics estimated on the uncontaminated
# record, and an alarm counts only where the uncontaminated run is silent. What
# comes out is the study's headline operational number: which movements this
# system finds, and how long each has to persist before it does.
#
# The three shapes are not arbitrary. A three-leaf stone wall — two masonry
# leaves either side of a weaker rubble-and-mortar core — fails in ways that
# leave distinguishable marks on a thermally driven inclination record:
#
# * **Amplitude growth.** The leaves stop acting together: delamination at the
#   core interface, or loss of through-stones. The section bends further under
#   the same daily heating, so the diurnal swing grows while its timing and its
#   mean stay put.
# * **Phase change.** The thermal path changes rather than the stiffness — water
#   entering the core raises its heat capacity, or a crack re-routes conduction.
#   The wall answers the same forcing later or earlier, which appears in the
#   residual as a harmonic in quadrature with the daily cycle. It is quoted as
#   the timing shift in hours, converted through the daily amplitude this
#   decomposition already measured.
# * **Drift.** Creep of the lime mortar under sustained load, thermal ratcheting
#   of the outer leaf, or foundation settlement: slow, monotone, invisible in any
#   single day, and quoted in millidegrees per year.
#
# The summer-2026 stretch Study 01 flagged is kept out of the reference window,
# but no claim is made about whether this detector finds it. Its size makes it
# uninformative about sensitivity, which is what this sweep exists to measure.

# %%
# The daily amplitude this decomposition fitted, which converts a timing shift
# into the residual amplitude it implies.
daily_amplitude = float(
    shares.set_index('component').loc['season_daily', 'peak_to_peak'] / 2.0)
phase_magnitudes = tuple(
    monitoring.phase_shift_amplitude(daily_amplitude, hours)
    for hours in DETECT_PHASE_SHIFTS_H)
print(f'Daily amplitude {daily_amplitude:.2f} mdeg; a timing shift of '
      f'{DETECT_PHASE_SHIFTS_H[0]} h to {DETECT_PHASE_SHIFTS_H[-1]} h implies '
      f'{phase_magnitudes[0]:.2f} to {phase_magnitudes[-1]:.2f} mdeg')

sweeps = {
    'amplitude': DETECT_MAGNITUDES,
    'phase': phase_magnitudes,
    'drift': DETECT_DRIFT_RATES,
}

curves = []
for kind in DETECT_KINDS:
    curve = monitoring.detectability_curve(
        in_control, reference_a['mu'], reference_a['sigma'],
        magnitudes=sweeps[kind], durations=DETECT_DURATIONS, kind=kind,
        freq=MODEL_FREQ_A, lam=EWMA_LAMBDA, L=EWMA_L, k=CUSUM_K, h=CUSUM_H,
        response_window=DETECT_RESPONSE_WINDOW)
    curves.append(curve.assign(kind=kind))

detectability = pd.concat(curves, ignore_index=True)
display(detectability)
detectability.to_csv(OUTPUT_DIR / 'NP_10_detectability.csv', index=False)

# %%
for kind in DETECT_KINDS:
    figures.plot_detectability(
        detectability[detectability['kind'] == kind],
        title=f'Detectability of a {kind} departure',
        save_path=str(OUTPUT_DIR), filename=f'NP_F09_detectability_{kind}')
    plt.show()

smallest = (detectability[detectability['detected']]
            .groupby(['kind', 'duration_h'])['magnitude'].min())
print('Smallest departure found, by mechanism and persistence:')
print(smallest.to_string() if len(smallest)
      else 'nothing detected anywhere in the swept range')

# %% [markdown]
# ## 8 · How far ahead is prediction worth anything?
#
# The target changes here, and so does the grid. The response is the gap-safe
# one-hour change: formed only between adjacent accepted observations inside one
# instrument era, so it never bridges a gap or the 2025 changeover. The level is
# not forecast, because its lag-one autocorrelation of 0.998 makes any error
# metric computed against it a measurement of the sampling interval.
#
# An absolute error at a given horizon answers nothing on its own, so every
# model is scored against three baselines a monitoring system could run for
# free: predicting no change at all, persisting the last change, and repeating
# the change from the same hour yesterday. Skill is the fractional reduction in
# mean absolute error against each, computed **paired** and with a block
# bootstrap, because residuals are autocorrelated and an unpaired comparison
# would report significance that is not there. A horizon counts as skilful only
# where the bootstrap interval excludes zero.
#
# The ladder exists because a model carrying both autoregressive memory and air
# temperature cannot say which of the two earned its skill. Each rung adds one
# thing; the increment is what that thing bought.
#
# ### Parameter Tuning Guidance
#
# **`MODEL_B_LAGS`** — autoregressive window in hours; default `24`, read off
# `NP_03`. Raising it costs training windows disproportionately on a fragmented
# record, because a segment shorter than `lags + forecasts` contributes nothing.
#
# **`MODEL_B_HORIZONS`** — horizons kept from the model output; default
# `(1, 3, 6, 12, 24, 48)`. Each must not exceed `MODEL_B_FORECASTS`.
#
# **`MODEL_B_REGRESSOR_LAGS`** — hours of predictor history; default `12`,
# inside study 03's admissible range for an external forcing. Study 03 found no
# level-band time constant that was not an artefact of the scan boundary, so no
# longer memory is justified.
#
# **`MODEL_B_SPECIFICATIONS`** — the ladder. Keep `'AR only'` first: without it
# nothing in this study is attributable.
#
# **`BOOTSTRAP_REPETITIONS`** — bootstrap draws; default `2000`. Lower it only
# for a smoke run; the reported intervals need the full count.

# %%
# The hourly grid. No era column is carried or consulted: instrument eras are
# out of scope for this study (D9), Study 01 having already compensated and
# anchored the record once across both, so hourly_change is called with
# era=None and never asked to break a difference at the changeover.
hourly = window.resample(MODEL_FREQ_B).mean(numeric_only=True)

frame_b = pd.DataFrame({
    'y': prediction.hourly_change(
        hourly[TARGET_COLUMN], era=None, freq=MODEL_FREQ_B),
    'tair': hourly['tair_str'],
    'rh': hourly['rh_str'],
    'batt': hourly['batt_str'],
})

segmented_b = prediction.contiguous_segments(
    frame_b, required=['y', 'tair', 'rh'],
    min_length=MODEL_B_MIN_SEGMENT, freq=MODEL_FREQ_B)
folds_b = prediction.expanding_segment_folds(
    segmented_b['segment_id'], MODEL_B_INITIAL_SEGMENTS, MODEL_B_FOLDS)
execution_b = prediction.execution_folds(
    folds_b, refit_each_fold=MODEL_B_REFIT_EACH_FOLD)

print(f'Model B: {len(segmented_b):,} rows in '
      f'{segmented_b["segment_id"].nunique()} segments, '
      f'{len(folds_b)} folds')

# %%
# backtest_specifications takes one dictionary per rung, carrying the rung's
# name and its regressors together with every setting the runner needs. The
# ladder itself stays a plain name-to-regressors mapping in the parameter cell,
# where a reader can see what each rung adds; it is expanded into the library's
# shape here. The Model B parameters are passed explicitly rather than left to
# the runner's defaults, which are Model A's and would silently cap the forecast
# at 24 hours.
specifications_b = [
    {'name': name, 'regressors': regressors, 'task': 'forecast',
     'n_lags': MODEL_B_LAGS, 'n_forecasts': MODEL_B_FORECASTS,
     'regressor_lags': MODEL_B_REGRESSOR_LAGS,
     'horizons': MODEL_B_HORIZONS, 'freq': MODEL_FREQ_B}
    for name, regressors in MODEL_B_SPECIFICATIONS.items()
]

predictions_b = prediction.backtest_specifications(
    segmented_b, execution_b, specifications_b,
    epochs=MODEL_B_EPOCHS, quantiles=MODEL_B_QUANTILES, seed=MODEL_B_SEED)
baselines_b = prediction.baseline_predictions(
    segmented_b, execution_b, MODEL_B_HORIZONS, task='forecast')
all_b = pd.concat([predictions_b, baselines_b], ignore_index=True)

naive_scale_b = float(segmented_b['y'].abs().mean())
metrics_b = prediction.score_predictions(
    all_b, ['model', 'horizon_h'], naive_scale=naive_scale_b,
    alpha=NOWCAST_INTERVAL_ALPHA)
display(metrics_b)

metrics_b.to_csv(OUTPUT_DIR / 'NP_11_forecast_metrics.csv', index=False)
figures.plot_metric_vs_horizon(
    metrics_b, metric='mae', by='model',
    title='Forecast error against horizon, by model and baseline',
    save_path=str(OUTPUT_DIR), filename='NP_F10_skill_vs_horizon')
plt.show()

# %%
# paired_mae_skill compares two absolute-error series indexed by prediction
# timestamp, and aligns them on the timestamps they share. The long prediction
# frames are therefore reduced to one such series per model and horizon here,
# once, rather than inside the loops below.
errors_b = {
    key: (block['y'] - block['yhat']).abs()
         .set_axis(pd.DatetimeIndex(block['ds'])).dropna()
    for key, block in all_b.groupby(['model', 'horizon_h'])
}

skill_rows = []
for baseline in ('zero', 'persistence', 'seasonal_naive'):
    for model in MODEL_B_SPECIFICATIONS:
        for horizon in MODEL_B_HORIZONS:
            parent = errors_b.get((baseline, horizon))
            child = errors_b.get((model, horizon))
            if parent is None or child is None:
                continue
            result = prediction.paired_mae_skill(
                parent, child, block_hours=BOOTSTRAP_BLOCK_HOURS,
                repetitions=BOOTSTRAP_REPETITIONS, seed=MODEL_B_SEED,
                horizon_hours=horizon)
            skill_rows.append({'baseline': baseline, 'model': model,
                               'horizon_h': horizon, **result})

skill = pd.DataFrame(skill_rows)
display(skill)
skill.to_csv(OUTPUT_DIR / 'NP_12_skill_vs_baseline.csv', index=False)

# The study's answer to its third question: the largest horizon at which the
# bootstrap interval for skill still excludes zero. Read against the hardest
# baseline the ladder's best rung was scored on.
print(f'Paired rows per comparison: {int(skill["n"].min()):,} to '
      f'{int(skill["n"].max()):,}')
if int(skill['n'].max()) == 0:
    raise RuntimeError(
        'No model and baseline share a single prediction timestamp, so every '
        'skill figure is missing rather than zero. An empty comparison is not '
        'a finding about the forecast and must not be reported as one.')

skilful = skill[(skill['skill_q05'] > 0.0) & (skill['model'] != 'AR only')]
for baseline, block in skilful.groupby('baseline'):
    best = block.loc[block['horizon_h'].idxmax()]
    print(f'vs {baseline}: skill excludes zero out to {best["horizon_h"]:.0f} h '
          f'({best["model"]}, skill {best["skill"]:.3f}, '
          f'90% interval {best["skill_q05"]:.3f} to {best["skill_q95"]:.3f})')
if skilful.empty:
    print('No model beats any baseline with an interval excluding zero at any '
          'horizon: on this record the forecast buys nothing a free baseline '
          'does not already deliver.')

# %%
# What each rung of the ladder bought, as the increment over the rung below it.
ladder = list(MODEL_B_SPECIFICATIONS)
ablation_rows = []
for lower, upper in zip(ladder[:-1], ladder[1:]):
    if upper.endswith('(control)'):
        continue
    for horizon in MODEL_B_HORIZONS:
        parent = errors_b.get((lower, horizon))
        child = errors_b.get((upper, horizon))
        if parent is None or child is None:
            continue
        result = prediction.paired_mae_skill(
            parent, child, block_hours=BOOTSTRAP_BLOCK_HOURS,
            repetitions=BOOTSTRAP_REPETITIONS, seed=MODEL_B_SEED,
            horizon_hours=horizon)
        ablation_rows.append({'added_over': lower, 'model': upper,
                              'horizon_h': horizon, **result})

ablation = pd.DataFrame(ablation_rows)
display(ablation)
ablation.to_csv(OUTPUT_DIR / 'NP_13_ablation.csv', index=False)

figures.plot_metric_vs_horizon(
    ablation.rename(columns={'model': 'rung'}), metric='skill', by='rung',
    title='Skill increment bought by each predictor, over the rung below it',
    save_path=str(OUTPUT_DIR), filename='NP_F11_ablation')
plt.show()

# %% [markdown]
# ## 9 · Can the model fill the gaps it was trained around?
#
# The operational temptation, once a model predicts a change, is to accumulate
# its predictions across a gap and call the result a reconstructed level. That
# is a stronger claim than anything measured so far: a per-step error that is
# small and unbiased still accumulates, and a reconstruction that arrives at the
# wrong level on the far side of a gap is worse than an admitted absence,
# because it looks like a measurement.
#
# The check is arithmetical rather than statistical. Where a gap is bracketed by
# accepted observations on both sides, the true change across it is known; the
# predicted changes are summed and compared. A reconstruction is safe only if it
# closes. Terminal gaps, and gaps crossing the instrument changeover, are
# reported with their status rather than scored.
#
# ### Parameter Tuning Guidance
#
# **`GAP_CLOSURE_TOLERANCE_MDEG`** — the median absolute closure error a
# reconstruction must stay below to be accepted. It is set against the smallest
# departure step 7b showed the monitor can detect: a reconstruction whose error
# exceeds what the monitor can see would manufacture alarms out of its own
# arithmetic.

# %%
prior_only = predictions_b[
    (predictions_b['model'] == 'AR + tair')
    & (predictions_b['horizon_h'] == 1)]

closure = prediction.gap_closure_summary(
    hourly[TARGET_COLUMN], prior_only.set_index('ds')['yhat'],
    era=None, freq=MODEL_FREQ_B)
display(closure)
closure.to_csv(OUTPUT_DIR / 'NP_14_gap_closure.csv', index=False)

# 'available' is the status gap_closure_summary gives a gap that is bracketed by
# accepted observations on both sides and has an estimate for every missing
# slot - the only kind whose closure can be scored at all. The rest carry
# 'unbracketed', 'cross_era' or 'incomplete_estimates' and are reported with
# their status rather than counted.
scorable = closure[closure['status'] == 'available']
unscorable = closure.loc[closure['status'] != 'available',
                         'status'].value_counts()
if len(unscorable):
    print(f'{len(scorable)} of {len(closure)} gaps are scorable; the rest are '
          + ', '.join(f'{count} {name}'
                      for name, count in unscorable.items()))
else:
    print(f'all {len(closure)} gaps are scorable')

if scorable.empty:
    verdict = 'undecidable - no gap is bracketed with complete estimates'
else:
    median_error = float(scorable['closure_error'].abs().median())
    verdict = ('safe to accumulate'
               if median_error < GAP_CLOSURE_TOLERANCE_MDEG
               else 'NOT safe to accumulate')
    print(f'Median absolute closure error {median_error:.2f} mdeg over '
          f'{len(scorable)} bracketed gaps, against a tolerance of '
          f'{GAP_CLOSURE_TOLERANCE_MDEG:.2f} mdeg')
print(f'Gap-closure verdict: reconstruction is {verdict}')

# %%
# One bracketed gap drawn whole: the observed level either side, and the level
# the accumulated predictions arrive at across it. Whether the reconstruction
# closes is visible rather than only tabulated.
if not scorable.empty:
    worst = scorable.loc[
        scorable['closure_error'].abs().sort_values(ascending=False).index]
    example = worst.iloc[0]
    span = slice(example['start'] - pd.Timedelta('48h'),
                 example['end'] + pd.Timedelta('48h'))
    observed_level = hourly[TARGET_COLUMN].loc[span]
    reconstructed = (observed_level.ffill().iloc[0]
                     + prior_only.set_index('ds')['yhat'].reindex(
                         observed_level.index).fillna(0.0).cumsum())

    print(f'Worst-closing bracketed gap {example["gap_id"]}: '
          f'{example["start"]} to {example["end"]}, observed recovery '
          f'{example["observed_recovery"]:+.2f} mdeg against predicted '
          f'{example["predicted_recovery"]:+.2f}, error '
          f'{example["closure_error"]:+.2f}')
    figures.plot_prediction_band(
        observed_level, reconstructed, reconstructed, reconstructed,
        freq=MODEL_FREQ_B,
        title='Accumulated prediction across the worst-closing bracketed gap',
        highlight=[(example['start'], example['end'])],
        save_path=str(OUTPUT_DIR), filename='NP_F12_gap_closure')
    plt.show()
else:
    print('No bracketed gap with complete estimates: NP_F12 is not drawn, and '
          'the closure question cannot be answered on this record.')

# %% [markdown]
# ## 10 · Run metadata and the report's table bodies
#
# Two exports that carry nothing new and exist so that nothing has to be
# retyped. The first records every parameter that governed this run, including
# the library versions, so a number in the report can be traced to the settings
# that produced it. The second writes one LaTeX body per table the report
# quotes; the report reads them with `\input`, so re-running this notebook
# updates the report's numbers and no figure in the prose can drift away from
# the artefact behind it.

# %%
import neuralprophet

metadata = pd.DataFrame([
    {'parameter': 'segment_start', 'value': SEGMENT_START},
    {'parameter': 'native_freq', 'value': NATIVE_FREQ},
    {'parameter': 'model_a_freq', 'value': MODEL_FREQ_A},
    {'parameter': 'model_b_freq', 'value': MODEL_FREQ_B},
    {'parameter': 'target_column', 'value': TARGET_COLUMN},
    {'parameter': 'predictors', 'value': ', '.join(PREDICTOR_COLUMNS)},
    {'parameter': 'model_a_lags', 'value': MODEL_A_LAGS},
    {'parameter': 'model_a_growth', 'value': MODEL_A_GROWTH},
    {'parameter': 'model_a_monitor_growth', 'value': MODEL_A_MONITOR_GROWTH},
    {'parameter': 'model_a_changepoints', 'value': MODEL_A_CHANGEPOINTS},
    {'parameter': 'model_a_yearly', 'value': MODEL_A_YEARLY},
    {'parameter': 'model_a_train_end', 'value': MODEL_A_TRAIN_END},
    {'parameter': 'model_a_epochs', 'value': MODEL_A_EPOCHS},
    {'parameter': 'rolling_refit_every', 'value': ROLLING_REFIT_EVERY},
    {'parameter': 'rolling_min_train', 'value': ROLLING_MIN_TRAIN},
    {'parameter': 'conformal_calibration_end',
     'value': CONFORMAL_CALIBRATION_END},
    {'parameter': 'model_b_lags', 'value': MODEL_B_LAGS},
    {'parameter': 'model_b_forecasts', 'value': MODEL_B_FORECASTS},
    {'parameter': 'model_b_regressor_lags', 'value': MODEL_B_REGRESSOR_LAGS},
    {'parameter': 'model_b_folds', 'value': MODEL_B_FOLDS},
    {'parameter': 'model_b_refit_each_fold', 'value': MODEL_B_REFIT_EACH_FOLD},
    {'parameter': 'model_b_epochs', 'value': MODEL_B_EPOCHS},
    {'parameter': 'seed', 'value': MODEL_A_SEED},
    {'parameter': 'quantiles', 'value': str(MODEL_A_QUANTILES)},
    {'parameter': 'reference_window',
     'value': f'{REFERENCE_START} to {REFERENCE_END}'},
    {'parameter': 'ewma_lambda', 'value': EWMA_LAMBDA},
    {'parameter': 'ewma_L', 'value': EWMA_L},
    {'parameter': 'cusum_k', 'value': CUSUM_K},
    {'parameter': 'cusum_h', 'value': CUSUM_H},
    {'parameter': 'joint_window', 'value': JOINT_WINDOW},
    {'parameter': 'target_arl_days', 'value': TARGET_ARL_DAYS},
    {'parameter': 'achieved_arl_days', 'value': round(float(chosen['arl_days']), 1)},
    {'parameter': 'gap_closure_tolerance_mdeg',
     'value': GAP_CLOSURE_TOLERANCE_MDEG},
    {'parameter': 'neuralprophet_version', 'value': neuralprophet.__version__},
    {'parameter': 'pandas_version', 'value': pd.__version__},
    {'parameter': 'numpy_version', 'value': np.__version__},
])
metadata.to_csv(OUTPUT_DIR / 'NP_15_run_metadata.csv', index=False)
display(metadata)

# %%
# LaTeX bodies for the tables the report quotes. Each column is given as a
# (source, format) pair, which is the specification tables.to_rows takes: the
# source is a column name or a callable on the row, and the format is a format
# string, a callable, or None for plain str.
tables.write_table(
    coverage.reset_index(), str(OUTPUT_DIR / 'NP_01_body.tex'),
    [('channel', tables.texttt), ('accepted', ',.0f'), ('coverage', '.1%')])

tables.write_table(
    gaps.groupby('gap_class', as_index=False)
        .agg(gaps=('n_slots', 'size'), hours=('duration_h', 'sum'))
        .sort_values('hours', ascending=False),
    str(OUTPUT_DIR / 'NP_02_body.tex'),
    [('gap_class', None), ('gaps', ',.0f'), ('hours', ',.1f')])

tables.write_table(
    survival, str(OUTPUT_DIR / 'NP_03_body.tex'),
    [('lag_hours', '.0f'), ('forecast_hours', '.0f'), ('n_segments', ',.0f'),
     ('n_surviving', ',.0f'), ('n_windows', ',.0f')])

tables.write_table(
    cadence, str(OUTPUT_DIR / 'NP_04_body.tex'),
    [('cadence', None), ('level_autocorr1', '.4f'), ('change_autocorr1', '.4f'),
     ('change_std', '.3f'), ('corr_change', '.4f'), ('drift_per_year', '.2f')])

tables.write_table(
    shares, str(OUTPUT_DIR / 'NP_05_body.tex'),
    [('component', tables.texttt), ('variance', ',.1f'), ('share', '.1%'),
     ('peak_to_peak', ',.1f')])

tables.write_table(
    gains, str(OUTPUT_DIR / 'NP_06_body.tex'),
    [('source', None), ('gain_mdeg_per_degC', '.2f'), ('method', None)])

tables.write_table(
    stability, str(OUTPUT_DIR / 'NP_16_body.tex'),
    [('block', None), ('tair_gain_mdeg_per_degC', '.2f'),
     ('trend_mdeg_per_year', '+.1f')])

tables.write_table(
    nowcast_scores, str(OUTPUT_DIR / 'NP_07_body.tex'),
    [('fit', None), ('n', ',.0f'), ('mae', '.2f'), ('rmse', '.2f'),
     ('bias', '+.2f'), ('mase', '.2f'), ('coverage_q05_q95', '.1%'),
     ('width_q05_q95', '.2f')])

tables.write_table(
    episodes_a, str(OUTPUT_DIR / 'NP_09_body.tex'),
    [(tables.date_cell('start', fmt='%Y-%m-%d %H:%M'), None),
     (tables.date_cell('end', fmt='%Y-%m-%d %H:%M'), None),
     ('duration_h', ',.1f'), ('mean_z', '+.2f'), ('peak_abs_z', '.2f')])

tables.write_table(
    detectability, str(OUTPUT_DIR / 'NP_10_body.tex'),
    [('kind', None), ('magnitude', '.2f'), ('duration_h', '.0f'),
     ('detected', tables.yes_no), ('delay_h', '.1f')])

tables.write_table(
    metrics_b, str(OUTPUT_DIR / 'NP_11_body.tex'),
    [('model', None), ('horizon_h', '.0f'), ('n', ',.0f'), ('mae', '.3f'),
     ('rmse', '.3f'), ('mase', '.3f'), ('coverage_q05_q95', '.1%')])

tables.write_table(
    skill, str(OUTPUT_DIR / 'NP_12_body.tex'),
    [('baseline', None), ('model', None), ('horizon_h', '.0f'),
     ('skill', '+.3f'), ('skill_q05', '+.3f'), ('skill_q95', '+.3f')])

tables.write_table(
    ablation, str(OUTPUT_DIR / 'NP_13_body.tex'),
    [('added_over', None), ('model', None), ('horizon_h', '.0f'),
     ('skill', '+.3f'), ('skill_q05', '+.3f'), ('skill_q95', '+.3f')])

tables.write_table(
    closure, str(OUTPUT_DIR / 'NP_14_body.tex'),
    [('gap_id', tables.texttt), (tables.date_cell('start'), None),
     ('n_missing', ',.0f'), ('status', None),
     ('observed_recovery', '+.2f'), ('predicted_recovery', '+.2f'),
     ('closure_error', '+.2f')])

tables.write_table(
    metadata, str(OUTPUT_DIR / 'NP_15_body.tex'),
    [('parameter', tables.texttt), ('value', None)])

print('Table bodies written:',
      ', '.join(sorted(p.name for p in OUTPUT_DIR.glob('*_body.tex'))))
