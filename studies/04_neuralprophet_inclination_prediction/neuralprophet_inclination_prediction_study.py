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

from shmlib import figures, prediction, proxies, site, viz

warnings.filterwarnings('ignore')
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)
viz.apply_report_style()

# %%
ARCHIVE_CSV = '../../data/interim/archive/gubbio_archive_20min.csv'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYSIS_FREQ = site.ANALYSIS_FREQ

# The response is Study 01's final inclination product: compensated, anchored
# once across the whole record, and cleaned of impulsive noise. It is read with
# honour_spike=True, so every value the cleaning interpolated is returned as
# missing rather than as a measurement. No earlier or intermediate version of
# this channel is used anywhere in this study.
TARGET_COLUMN = 'inc_comp_cleaned'

# The window. The archive carries a 271-day outage that begins on 2022-09-23 and
# whose last missing day is 2023-06-20, as recorded in
# docs/data-quality-report-2026-08-10.md. This study starts on the day after it
# and runs to the end of the archive. The cell below reports the first
# inclination value actually accepted inside the window, which is what the
# report quotes.
SEGMENT_START = '2023-06-21'

# The variables the instrument package on the wall records, in the order their
# panels are drawn: the response first, then the thermal channels, then the
# rest. Supply voltage is recorded too, but it is instrument housekeeping rather
# than a measurement of the wall or of its environment, so it is left out of the
# record figure.
ON_STRUCTURE_COLUMNS = (
    TARGET_COLUMN, 'twall_str', 'tair_str', 'sr_str', 'rh_str')

# The archive names the package's channels differently before and after the
# instrument installed on 2025-02-21, so each set is loaded under its own map
# and the two are joined into one set of channels. This is a column-naming
# detail and not an analytical split: Study 01 already compensated and anchored
# the inclination once across the whole record.
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
    freq=ANALYSIS_FREQ, tz=site.SITE_TZ)

sensor_current, sensor_provenance = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_CURRENT, freq=ANALYSIS_FREQ,
    tz=site.SITE_TZ, honour_suspect=True)
sensor_legacy, _ = proxies.load_sensor_forcings(
    ARCHIVE_CSV, column_map=STR_MAP_LEGACY, freq=ANALYSIS_FREQ,
    tz=site.SITE_TZ, honour_suspect=True)
sensor = proxies.join_eras([sensor_current, sensor_legacy])

record = proxies.harmonise(
    [sensor, inclination.to_frame()], freq=ANALYSIS_FREQ)
window = record.loc[site.to_utc(pd.DatetimeIndex([SEGMENT_START]))[0]:]

observed = window[TARGET_COLUMN].dropna()
print(f'Window: {window.index.min()} to {window.index.max()} '
      f'({len(window):,} slots on the {ANALYSIS_FREQ} grid)')
print(f'First accepted inclination inside the window: {observed.index.min()}')

# Coverage per channel over the window, which is what the report quotes. It
# counts accepted values against the slots of the window, and says nothing yet
# about how the missing time is distributed — that is the next movement.
coverage = pd.DataFrame({
    'accepted': window[list(ON_STRUCTURE_COLUMNS)].notna().sum(),
    'coverage': window[list(ON_STRUCTURE_COLUMNS)].notna().mean(),
})
coverage.index.name = 'channel'
print(coverage.to_string(formatters={'coverage': '{:.1%}'.format}))

# %%
figures.plot_channel_panels(
    window, list(ON_STRUCTURE_COLUMNS),
    title='The on-structure record after the 2022-2023 outage',
    save_path=str(OUTPUT_DIR), filename='NP_F01_on_structure_record')
plt.show()
