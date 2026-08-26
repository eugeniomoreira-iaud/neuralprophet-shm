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

from shmlib import adc, figures, prediction, proxies, site, tables, viz

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

# The raw channel, spike-masked, used once as a sensitivity in step 5: fitting
# the thermal term rather than subtracting the documented coefficient is the
# only way this study can speak to the sign contradiction of
# docs/raw-data-format.md section 7.5.
RAW_TARGET_COLUMN = 'inc'

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
cadence = prediction.cadence_evidence(
    window[TARGET_COLUMN], window['tair_str'], era=window.get('era'),
    cadences=(MODEL_FREQ_A, MODEL_FREQ_B), freq=NATIVE_FREQ)
display(cadence)

cadence.to_csv(OUTPUT_DIR / 'NP_04_cadence_evidence.csv', index=False)
figures.plot_cadence_evidence(
    cadence, title='What fixes the cadence and the target',
    save_path=str(OUTPUT_DIR), filename='NP_F04_cadence_evidence')
plt.show()
