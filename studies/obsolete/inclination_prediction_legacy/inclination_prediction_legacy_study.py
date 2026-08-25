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
# # Study · Predicting the inclination from the legacy sensor set
#
# The 14-column network ran at station 02 from 26 July 2018 until the changeover
# on 21 February 2025 — six and a half years, against the 143 usable days of the
# extended package. It carries only air temperature, relative humidity, battery
# voltage and inclination: no solar radiation, no wall temperature.
#
# This study asks two families of question. The first is what the extra length
# buys, and is specific to this era. The second is the set the extended-sensor
# study asked, repeated here so that the two eras are directly comparable —
# every test below uses the sibling study's machinery unchanged, so any
# difference in result is a difference in the data and not in the method.
#
# ## The questions
#
# **Specific to the long record**
#
# 1. **Which periodicities are present**, established on gapped data without
#    imputing anything first?
# 2. **Over which stretch should they be decomposed**, and do they persist?
# 3. **How should missing values be filled**, and up to what gap length is that
#    defensible?
#
# **Ported from `../inclination_prediction/`, for comparability**
#
# 4. **Can one sensor channel predict the inclination, and which is best?**
# 5. **Can a combination beat the best single channel, and how is the best
#    combination diagnosed?**
# 6. **How far ahead can NeuralProphet forecast, with what uncertainty — and
#    what is the error at each horizon actually made of?**
#
# Question 6 merges the sibling study's horizon question with this study's error
# budget, which is the natural place for it: a horizon curve says how large the
# error is, and the budget says which component produced it.
#
# **Raised by the answers, and added afterwards**
#
# 7. **How much of the record can NeuralProphet actually use**, once the demand
#    for a continuous series is dropped — and what does each way of extending it
#    cost in fabricated data?
#
# Question 7 exists because Questions 2 and 6 were answered under a restriction
# that Step 2 asserted rather than tested: that autoregressive work must run on
# one unbroken block. Step 13b tests it.
#
# ## Premises
#
# Unchanged from the sibling study and not re-tested here:
#
# | Premise | Consequence |
# |---|---|
# | The logged `I` channel is an inclination in millidegrees | The 2500 offset is added back. In this era the logged values run 2105–2300, so read as millidegrees they lie **outside** the ±2000 mdeg range the datasheet certifies. That is accepted by instruction; the compensation normalises the series to start at zero, so nothing downstream depends on it. |
# | The documented compensation is correct | `inc_comp` is the target of every model. |
# | The analysis grid is one hour | Matching every external proxy the pipeline aligns against. |
#
# The site's sign convention and the 12-hour bound on lags for external forcings
# are as recorded in `docs/raw-data-format.md` Section 7.5.
#
# ## Steps
#
# 1. **Load** the legacy era, station 02, hourly.
# 2. **Coverage, gaps and blocks** — what the record can and cannot support.
# 3. **Description** and the expected sign, tested on the raw channel.
# 4. **Spectrum** — Lomb-Scargle with the sampling-pattern control and a
#    permutation null. *(Question 1)*
# 5. **Harmonic decomposition and its stability**. *(Question 2)*
# 6. **Imputation**, measured against injected gaps. *(Question 3)*
# 7. **Operators** — full band against diurnal band, then a signed-delay
#    control. *(Question 4, core)*
# 8. **Ranking** the single predictors, on two windows. *(Question 4)*
# 9. **Combinations** — collinearity, exhaustive search, leave-one-out gain,
#    permutation importance, out-of-period test. *(Question 5)*
# 10. **The diagnostic protocol.**
# 11. **Error budget** — the nested ladder across eleven horizons. *(Question 6)*
# 12. **NeuralProphet** — horizons, regimes, calibration, and what the joint
#     multi-horizon head costs. *(Question 6)*
# 13. **Sensitivity.**
# 13b. **How much of the record NeuralProphet can actually use** — the gap
#     tolerance sweep, windows gained against hours invented, whether the record
#     is one series or two, and the decomposition and prediction regimes that the
#     longest-block restriction had been hiding. *(Question 7)*
# 14. **Verdicts.**

# %% [markdown]
# ## Imports and configuration

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

sys.path.insert(0, os.path.abspath('.'))       # lp_lib, local to this study
sys.path.insert(0, os.path.abspath('../../..'))   # heritageshm, at the repo root

import lp_lib as lp
import ip_lib as ip
import tc_lib as tc
import lc_lib as lc

warnings.filterwarnings('ignore')
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 80)

# %% [markdown]
# ### Parameters
#
# | Parameter | Purpose |
# |---|---|
# | `ARCHIVE_DIR` | Read-only `.adc` archive. Never written to. |
# | `CACHE_DIR` | Local working copy of the 1926 legacy files. |
# | `ERA` | Bounds of the 14-column era. |
# | `MODEL_WINDOW` | Longest unbroken stretch, used for the autoregressive work. |
# | `HOLDOUT` | A later block, held back entirely for the out-of-period test. |
# | `PERIOD_*` | Trial-period grids for the two spectral bands. |
# | `N_PERMUTATIONS` | Replicates setting the periodogram's false-alarm level. |
# | `DETREND_HOURS` | Rolling mean removed before the high-frequency spectrum. |
# | `SLIDE_*` | Width and step of the moving harmonic fit. |
# | `DELAYS` / `SIGNED_DELAYS` | Causal and signed transport-delay grids. |
# | `HORIZONS_LADDER` | Horizons for the error budget, out to one year. |
# | `HORIZONS_NP` | Horizons for NeuralProphet, which cannot span a year. |
# | `IMPUTE_TOLERANCE` | Error accepted from a filled value, in mdeg. |
# | `SEGMENT_FILL_H` | Longest interruption Step 13b interpolates across. |

# %%
ARCHIVE_DIR = os.path.expanduser(
    '~/Library/CloudStorage/GoogleDrive-eugeniomoreira@iaud.ufc.br/'
    'My Drive/_UNIPG/__Mura-realtime'
)
CACHE_DIR  = '.cache'
OUTPUT_DIR = 'outputs'
STATION    = 'st02'

ANALYSIS_FREQ  = '1h'
CONTEXT        = 'notebook'    # 'paper' re-exports every figure at column size
SAMPLING_HOURS = 1.0
ERA            = lp.LEGACY_ERA

TARGET  = 'inc_comp'
DRIVERS = lp.LEGACY_DRIVERS            # tair, rh
CONTROL = lp.CONTROL_DRIVER            # batt, negative control

PERIOD_LOW     = lp.period_grid(2, 1500, n=3000)
PERIOD_HIGH    = lp.period_grid(0.2, 10, n=2000)
N_PERMUTATIONS = 200
DETREND_HOURS  = 168
DETREND_CHECK  = 720           # second width, to expose filter artefacts

SLIDE_WINDOW_DAYS = 730        # two years: the minimum that identifies an annual term
SLIDE_STEP_DAYS   = 90

# The longest unbroken stretch, from the block report in Step 2, and a later
# block held back entirely. No block in this record contains two full annual
# cycles, which is itself a finding.
MODEL_WINDOW = ('2020-07-31', '2022-04-09')
HOLDOUT      = ('2023-06-21', '2024-05-04')

HORIZONS_LADDER = [1, 3, 6, 12, 24, 72, 168, 720, 2160, 4380, 8760]
HORIZONS_NP     = [1, 3, 6, 12, 24, 72, 168]
LADDER_ORIGINS  = (0.5, 0.6, 0.7)
N_LAGS          = 24

IMPUTE_TOLERANCE  = 5.0
IMPUTE_GAPS       = (1, 3, 6, 12, 24, 72, 168, 720)
IMPUTE_TRIALS     = 40

# Transport delay bounded at 12 h for the external forcings, per docs Section
# 7.5. The signed grid is diagnostic only: `causal_only` keeps any lead out of
# every predictive model.
MAX_DELAY_H   = 12
DELAYS        = list(range(0, MAX_DELAY_H + 1))
SIGNED_DELAYS = list(range(-MAX_DELAY_H, MAX_DELAY_H + 1))
TAUS          = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168]

CV_SPLITS      = 5
MIN_TRAIN_FRAC = 0.5
RIDGE_ALPHA    = 1.0
N_REPEATS      = 10

# Step 13b fits across every complete stretch of the era rather than the longest
# block alone. Only interruptions this short are interpolated, which is the limit
# the Step 6 benchmark licenses for interpolation; everything longer splits the
# record into another stretch and is never invented.
SEGMENT_FILL_H = 6

RUN_NEURALPROPHET = True
NP_EPOCHS         = 30
NP_TRAIN_FRAC     = 0.70
QUANTILES         = (0.05, 0.95)

os.makedirs(OUTPUT_DIR, exist_ok=True)
lp.set_context(CONTEXT)

# %% [markdown]
# ## Step 1 · Load the legacy era
#
# The legacy parser tolerates the defects concentrated in this era, which the
# quality report measured: a third of files carry duplicate timestamps whose
# copies disagree, an eighth mix decimal separators internally, and some carry
# records dated to the previous day. Duplicates are merged field by field rather
# than dropped, because where copies differ it is usually because one carries
# real values where the other carries zeros.

# %%
df = lp.load_station(ARCHIVE_DIR, CACHE_DIR, ERA[0], ERA[1],
                     station=STATION, freq=ANALYSIS_FREQ)
df, flagged = tc.screen_spikes(df, col='inc', window=3, k=8.0)
df = ip.add_target(df)
print(f'grid: {len(df)} hourly slots, {df.index.min()} to {df.index.max()}')
df.head()

# %%
summary = df[['inc', TARGET, 'tair', 'rh', CONTROL]].describe().T.round(3)
display(summary)
summary.to_csv(f'{OUTPUT_DIR}/S4_01_{STATION}_channel_summary.csv')

print(f'\nlogged inclination runs {df["inc"].min():.1f} to {df["inc"].max():.1f} '
      f'mdeg.\nRead as millidegrees this is outside the +/- 2000 mdeg the '
      f'datasheet certifies, which is accepted by instruction.')

# %% [markdown]
# ## Step 2 · Coverage, gaps, and what the record can support

# %%
coverage = tc.coverage_table(df)
display(coverage)
coverage.to_csv(f'{OUTPUT_DIR}/S4_02_{STATION}_coverage.csv')

# %%
inventory = lp.gap_inventory(df[TARGET], dt_hours=SAMPLING_HOURS)
display(inventory)
print(f'{inventory.attrs["n_gaps"]} gaps in all; the longest is '
      f'{inventory.attrs["longest_hours"]:.0f} h '
      f'({inventory.attrs["longest_hours"] / 24:.0f} days)')
inventory.to_csv(f'{OUTPUT_DIR}/S4_03_{STATION}_gap_inventory.csv')

# %% [markdown]
# ### Which blocks could carry an annual decomposition
#
# A component can only be identified from a stretch that contains it several
# times over. The rule is applied explicitly rather than by inspection, at
# several gap tolerances, because how much interruption a block may absorb is
# exactly what the imputation step of Step 6 decides.

# %%
block_tables = {}
for tol in (6, 24, 72):
    wr = lp.window_report(df, [TARGET] + DRIVERS, lp.YEAR_DAYS,
                          min_cycles=2.0, min_days=60, max_gap_hours=tol)
    block_tables[tol] = wr
    if len(wr):
        best = wr.iloc[0]
        print(f'gap tolerance {tol:>3} h: {len(wr)} blocks >= 60 d; longest '
              f'{best["days"]:.0f} d = {best["cycles"]:.2f} annual cycles, '
              f'qualifies: {bool(best["qualifies"])}')
    else:
        print(f'gap tolerance {tol:>3} h: no block of 60 days or more')
display(block_tables[72].head(8))
block_tables[72].to_csv(f'{OUTPUT_DIR}/S4_04_{STATION}_blocks.csv', index=False)

# %% [markdown]
# **No block in this record contains two full annual cycles.** The longest
# unbroken stretch, even absorbing interruptions of up to three days, is under
# 1.7 cycles. The annual term therefore cannot be identified from any single
# contiguous window, and must be fitted across the gaps — which is precisely what
# a least-squares harmonic fit does and what an autoregressive model cannot.
#
# That divides the study. The **spectrum and the harmonic decomposition run on
# the whole era**, because they tolerate gaps by construction. The
# **autoregressive work runs on the longest block**, because a single fitted
# window does not. A later block is held back entirely for the out-of-period test
# in Step 9.
#
# The second half of that sentence is weaker than it looks, and **Step 13b takes
# it apart**. An autoregressive model does not need one continuous series, only
# windows of consecutive observations, and those exist in every block. Everything
# from here to Step 13 keeps the restriction so that the results stay comparable
# with the sibling study; Step 13b then measures what it costs.

# %%
model_df = df.loc[MODEL_WINDOW[0]:MODEL_WINDOW[1]].copy()
holdout_df = df.loc[HOLDOUT[0]:HOLDOUT[1]].copy()
for name, w in [('full era   ', df), ('model block', model_df),
                ('holdout    ', holdout_df)]:
    print(f'{name}: {w.index.min().date()} to {w.index.max().date()} | '
          f'{len(w)} slots | {w[TARGET].notna().mean():.1%} observed')

# %% [markdown]
# ## Step 3 · What the signal looks like

# %%
tc.plot_overview(
    df, cols=('inc_comp', 'inc', 'tair', 'rh', CONTROL),
    title='Legacy era, station 02 — compensated inclination and its channels',
    save_path=OUTPUT_DIR, filename=f'S4_F01_{STATION}_overview')
plt.show()

# %%
tc.plot_compensation_effect(
    df, df[TARGET], temp_col='tair',
    title='Raw and compensated inclination over six and a half years',
    save_path=OUTPUT_DIR, filename=f'S4_F02_{STATION}_target')
plt.show()

# %% [markdown]
# ### The expected sign, tested on the raw channel
#
# The site predicts a negative association between the inclination and any
# heating driver. As in the extended-sensor study, that is tested on the raw
# channel before the compensation can manufacture the expected sign.

# %%
signs = ip.sign_table(df, ['inc', TARGET], DRIVERS,
                      detrend_hours=DETREND_HOURS, dt_hours=SAMPLING_HOURS)
display(signs)
signs.to_csv(f'{OUTPUT_DIR}/S4_05_{STATION}_sign_convention.csv')

# %% [markdown]
# ### Correlation structure

# %%
corr_cols = [TARGET, 'inc'] + DRIVERS + [CONTROL]
tc.plot_correlation_heatmaps(
    df, corr_cols, title='Correlation structure, legacy era',
    save_path=OUTPUT_DIR, filename=f'S4_F09_{STATION}_correlations')
plt.show()

corr_levels = tc.correlation_matrix(df, corr_cols, differenced=False)
corr_diff = tc.correlation_matrix(df, corr_cols, differenced=True)
corr_levels.round(3).to_csv(f'{OUTPUT_DIR}/S4_06_{STATION}_corr_levels.csv')
corr_diff.round(3).to_csv(f'{OUTPUT_DIR}/S4_07_{STATION}_corr_diff.csv')
display(corr_levels.round(3))

# %% [markdown]
# ## Step 4 · The spectrum
#
# **Question 1.** A fifth of the record is missing, and the missingness is
# structured into a handful of long outages rather than scattered. That rules out
# an ordinary periodogram, which needs a complete grid: filling those holes
# before estimating the spectrum lets the filling method decide the answer, and a
# 271-day hole interpolated linearly injects power at exactly the low frequencies
# under investigation.
#
# The Lomb-Scargle periodogram fits a sinusoid at each trial frequency by least
# squares over whatever samples exist, so it consumes the gaps rather than
# requiring them filled. **The spectrum is therefore computed before any
# imputation**, and Step 6 cannot influence it.
#
# Two controls accompany it. The **window function** is the same periodogram run
# on the observation mask alone: the outage pattern has its own spectrum, and any
# peak appearing there is a property of when the instrument was running. The
# **permutation null** shuffles the values against the observation times, which
# destroys periodicity while preserving the sampling pattern exactly.

# %%
daily = df[TARGET].resample('1D').mean()
spec_low, levels = lp.lomb_scargle(daily, PERIOD_LOW,
                                   n_bootstrap=N_PERMUTATIONS, seed=0)
window_low = lp.window_spectrum(daily, PERIOD_LOW)
peaks_low = lp.spectral_peaks(spec_low, levels, window_low, top=10)
display(peaks_low)
peaks_low.to_csv(f'{OUTPUT_DIR}/S4_08_{STATION}_peaks_low.csv', index=False)
print(f'permutation levels: {levels}')

# %%
lp.plot_spectrum(
    spec_low, window=window_low, levels=levels,
    marks={'1 yr': lp.YEAR_DAYS, '0.5 yr': lp.YEAR_DAYS / 2},
    title='Low-frequency spectrum of the compensated inclination, daily means',
    save_path=OUTPUT_DIR, filename=f'S4_F03_{STATION}_spectrum_low')
plt.show()

# %% [markdown]
# ### The diurnal band
#
# Normalised power is a share of total variance, and the annual term holds so
# much of it that the daily cycle would sit in the fourth decimal place. The
# high-frequency spectrum is therefore computed on the band-limited series.
#
# That filter has a cost, and the second width below exposes it: a rolling mean
# of width `W` acts as a high-pass whose response peaks near `W` itself, so a
# feature that moves when `W` changes is the filter, not the structure.

# %%
spec_high, _ = lp.lomb_scargle(df[TARGET], PERIOD_HIGH,
                               detrend_hours=DETREND_HOURS)
peaks_high = lp.spectral_peaks(spec_high, top=6)
display(peaks_high)
peaks_high.to_csv(f'{OUTPUT_DIR}/S4_09_{STATION}_peaks_high.csv', index=False)

# %%
spec_high_check, _ = lp.lomb_scargle(df[TARGET], PERIOD_HIGH,
                                     detrend_hours=DETREND_CHECK)
peaks_check = lp.spectral_peaks(spec_high_check, top=6)
print(f'detrend width {DETREND_HOURS} h -> peaks at '
      f'{list(peaks_high["label"][:4])}')
print(f'detrend width {DETREND_CHECK} h -> peaks at '
      f'{list(peaks_check["label"][:4])}')
print('Any peak that moves with the filter width is an artefact of the filter.')

# %%
lp.plot_spectrum(
    spec_high, marks={'24 h': 1.0, '12 h': 0.5, '8 h': 1 / 3},
    title=f'Diurnal band, {DETREND_HOURS} h rolling mean removed',
    save_path=OUTPUT_DIR, filename=f'S4_F04_{STATION}_spectrum_high')
plt.show()

# %% [markdown]
# ### Welch cross-check on an unbroken stretch

# %%
psd = lp.welch_psd(model_df[TARGET], freq_hours=SAMPLING_HOURS,
                   segment_days=60)
band = psd.loc[0.5:3.0]
print(f'Welch peak in the daily band: {band["psd"].idxmax():.4f} d '
      f'({band["psd"].idxmax() * 24:.2f} h)')

# %% [markdown]
# ## Step 5 · Harmonic decomposition, and whether it holds still
#
# **Question 2.** One limit has to be stated before any semi-annual result is
# read. Separating a **180.0-day** component from the annual second harmonic at
# **182.62 days** requires resolving a frequency difference of
# 8.0 x 10⁻⁵ cycles per day, which needs a record of roughly **34 years**. With
# six and a half, it cannot be done, and no choice of estimator changes that.
#
# Phase behaviour can separate them. If the semi-annual term is harmonic
# distortion of the annual cycle — which any nonlinear response to a sinusoidal
# annual driver produces automatically — then its phase advances at exactly twice
# the annual rate, and `2·φ_annual − φ_semiannual` stays constant along the
# record. If it is an independent process, that combination drifts.

# %%
harm_table, harm_fit, harm_model = lp.fit_harmonics(
    df[TARGET], lp.HARMONIC_PERIODS)
display(harm_table)
print(f'R2 of the full harmonic model: {harm_table.attrs["r2"]:.4f}')
harm_table.to_csv(f'{OUTPUT_DIR}/S4_10_{STATION}_harmonics.csv')

# %%
fig, ax = plt.subplots(figsize=lp.figsize(11, 3.2))
tc._ts(ax, df[TARGET], label='observed', lw=0.6, color='0.35')
tc._ts(ax, harm_fit, label='harmonic model', lw=1.0)
ax.set_ylabel('compensated inclination [mdeg]')
ax.set_xlabel('')
ax.legend(fontsize='small')
ax.set_title('Harmonic model fitted across the gaps, whole era')
tc._finish(fig, OUTPUT_DIR, f'S4_F05_{STATION}_harmonic_fit')
plt.show()

# %%
sliding = lp.sliding_harmonics(
    df[TARGET], {k: lp.HARMONIC_PERIODS[k] for k in ('annual', 'semiannual')},
    window_days=SLIDE_WINDOW_DAYS, step_days=SLIDE_STEP_DAYS)
stability = sliding.groupby('component')[
    ['amplitude_mdeg', 'peak_day_after_t0']].agg(['mean', 'std', 'min', 'max'])
display(stability.round(2))
sliding.to_csv(f'{OUTPUT_DIR}/S4_11_{STATION}_sliding_harmonics.csv',
               index=False)
stability.round(3).to_csv(f'{OUTPUT_DIR}/S4_12_{STATION}_harmonic_stability.csv')

# %%
lock = lp.phase_lock_test(sliding)
print('phase-lock test, semi-annual against annual:')
for k, v in lock.items():
    print(f'  {k}: {v}')
pd.Series(lock).to_frame('value').to_csv(
    f'{OUTPUT_DIR}/S4_13_{STATION}_phase_lock.csv')

# %%
lp.plot_sliding_harmonics(
    sliding,
    title=f'Component amplitude and phase along the record '
          f'({SLIDE_WINDOW_DAYS} d windows)',
    save_path=OUTPUT_DIR, filename=f'S4_F06_{STATION}_sliding_harmonics')
plt.show()

# %%
tc.plot_diurnal(
    df, cols=(TARGET, 'tair', 'rh'),
    title='Mean diurnal cycle, whole legacy era',
    save_path=OUTPUT_DIR, filename=f'S4_F07_{STATION}_diurnal')
plt.show()

# %% [markdown]
# ## Step 6 · Imputation, measured rather than asserted
#
# **Question 3.** Filling a hole is defensible only up to the length at which the
# filler stops knowing anything the data did not already contain. That length is
# measurable: remove stretches whose values are known, fill them, and compare.

# %%
bench = lp.impute_benchmark(
    df, TARGET, DRIVERS, lp.HARMONIC_PERIODS, gap_lengths=IMPUTE_GAPS,
    n_trials=IMPUTE_TRIALS, seed=0, min_context_hours=48)
display(bench.pivot(index='gap_hours', columns='method', values='MAE_mdeg'))
bench.to_csv(f'{OUTPUT_DIR}/S4_14_{STATION}_imputation_benchmark.csv',
             index=False)

# %%
policy = lp.impute_policy(bench, tolerance_mdeg=IMPUTE_TOLERANCE)
display(policy)
policy.to_csv(f'{OUTPUT_DIR}/S4_15_{STATION}_imputation_policy.csv')
print(f'Tolerance applied: {IMPUTE_TOLERANCE} mdeg, against a signal standard '
      f'deviation of {df[TARGET].std():.1f} mdeg.')

# %%
lp.plot_gap_inventory(
    inventory, bench=bench,
    title='Where the missing hours are, and what filling them costs',
    save_path=OUTPUT_DIR, filename=f'S4_F08_{STATION}_gaps_imputation')
plt.show()

# %% [markdown]
# ## Step 7 · Operators — full band against diurnal band
#
# **Question 4, core.** Before any driver is ranked it is given the chance to act
# through the operator that suits it: a **transport delay**, which shifts the
# driver without changing its shape, and a **thermal inertia**, which low-passes
# it with a time constant. The delay is bounded at twelve hours because both
# candidates here are external forcings, which must causally precede the response
# (`docs/raw-data-format.md` Section 7.5).
#
# The scan is run on both bands, and the comparison is the point. On the full
# band an optimum that runs to the edge of the τ grid is fitting the seasonal
# envelope rather than a thermal time constant.

# %%
ops_full, scans_full = ip.operator_table(
    df, TARGET, DRIVERS + [CONTROL], DELAYS, TAUS,
    dt_hours=SAMPLING_HOURS, detrend_hours=None)
display(ops_full)
ops_full.to_csv(f'{OUTPUT_DIR}/S4_16_{STATION}_operators_fullband.csv')

# %%
ip.plot_operator_heatmaps(
    {k: scans_full[k] for k in DRIVERS},
    title='Delay against inertia, full band — note any optimum at the τ edge',
    save_path=OUTPUT_DIR, filename=f'S4_F10_{STATION}_operators_fullband')
plt.show()

# %%
ops_band, scans_band = ip.operator_table(
    df, TARGET, DRIVERS + [CONTROL], DELAYS, TAUS,
    dt_hours=SAMPLING_HOURS, detrend_hours=DETREND_HOURS)
display(ops_band)
ops_band.to_csv(f'{OUTPUT_DIR}/S4_17_{STATION}_operators_diurnal.csv')
OPERATORS = ip.operators_from_summary(ops_band, dt_hours=SAMPLING_HOURS,
                                      causal_only=True, verbose=True)

# %%
ip.plot_operator_heatmaps(
    {k: scans_band[k] for k in DRIVERS},
    title='Delay against inertia, diurnal band — carried forward',
    save_path=OUTPUT_DIR, filename=f'S4_F11_{STATION}_operators_diurnal')
plt.show()

# %% [markdown]
# ### The signed-delay control — a diagnostic, not a model input
#
# **Nothing in this section feeds any model.** The operators used from here on
# are the causal ones fixed above.
#
# The extended-sensor study found that wall temperature — an *internal state
# variable* — legitimately lags the deformation, and had to be scanned over
# signed delays. This era has no wall probe: both candidates are **external
# forcings**, and a forcing must precede the response it causes. So the signed
# scan here is a **control rather than a measurement**. If either driver peaked
# at a negative delay, the method would be at fault, not the physics.

# %%
ops_signed, scans_signed = ip.operator_table(
    df, TARGET, DRIVERS, SIGNED_DELAYS, TAUS,
    dt_hours=SAMPLING_HOURS, detrend_hours=DETREND_HOURS)
ops_signed['causal_r2'] = ops_band.loc[ops_signed.index, 'r2']
ops_signed['r2_lost_to_causality'] = (
    ops_signed['r2'] - ops_signed['causal_r2']).round(4)
display(ops_signed)
ops_signed.to_csv(f'{OUTPUT_DIR}/S4_18_{STATION}_operators_signed.csv')

# A negative optimum only matters if it buys something. On a grid this fine an
# optimum can land one step to the left of zero on noise alone, so the control
# tests the *gain* rather than the sign: a lead counts as a violation only if
# relaxing causality wins more than this much explained variance.
LEAD_TOLERANCE_R2 = 0.01

leads = ops_signed[ops_signed['delay_h'] < 0]
material = leads[leads['r2_lost_to_causality'] > LEAD_TOLERANCE_R2]
controls_ok = bool(material.empty)
print(f'control check — no external forcing gains more than '
      f'{LEAD_TOLERANCE_R2} R2 from a lead: {controls_ok}')
for drv, row in leads.iterrows():
    print(f'  {drv}: optimum at {row["delay_h"]:+.0f} h, buying '
          f'{row["r2_lost_to_causality"]:+.4f} R2 — '
          f'{"MATERIAL" if drv in material.index else "within noise"}')
if not controls_ok:
    print('  WARNING: a forcing materially leads the response. Treat the '
          'operator estimates as a method artefact, not a measurement.')

# %%
phase = ip.cross_phase(df, [(TARGET, d) for d in DRIVERS]
                       + [('inc', 'tair')], SIGNED_DELAYS,
                       detrend_hours=DETREND_HOURS, dt_hours=SAMPLING_HOURS)
display(phase)
phase.to_csv(f'{OUTPUT_DIR}/S4_19_{STATION}_phase_control.csv')

# %%
ip.plot_operator_heatmaps(
    {k: scans_signed[k] for k in DRIVERS},
    title='Signed delays — control only, feeds no model',
    save_path=OUTPUT_DIR, filename=f'S4_F12_{STATION}_operators_signed')
plt.show()

# %% [markdown]
# ## Step 8 · Ranking the single predictors
#
# **Question 4.** Each driver is scored on its own, at its diurnal-band operator,
# by pooled error over five rolling-origin folds. Autoregression is deliberately
# excluded: with it, everything ties behind the autoregressive term at short
# horizons and the ranking says nothing. Battery voltage is carried as a
# **negative control**.

# %%
rank = ip.rank_single_predictors(
    df, TARGET, DRIVERS + [CONTROL], OPERATORS, dt_hours=SAMPLING_HOURS,
    n_splits=CV_SPLITS, min_train_frac=MIN_TRAIN_FRAC, alpha=RIDGE_ALPHA,
    full_record=df)
display(rank)
rank.to_csv(f'{OUTPUT_DIR}/S4_21_{STATION}_rank_full.csv')

# %%
ip.plot_single_ranking(
    rank, title='Single-predictor skill, whole era (no autoregression)',
    save_path=OUTPUT_DIR, filename=f'S4_F13_{STATION}_rank')
plt.show()

# %% [markdown]
# ### The same ranking on the unbroken block
#
# If a channel's rank survives the change of window it is a property of the
# structure; if it does not, it was a property of the period.

# %%
rank_block = ip.rank_single_predictors(
    model_df, TARGET, DRIVERS + [CONTROL], OPERATORS,
    dt_hours=SAMPLING_HOURS, n_splits=CV_SPLITS,
    min_train_frac=MIN_TRAIN_FRAC, alpha=RIDGE_ALPHA, full_record=df)
display(rank_block)
rank_block.to_csv(f'{OUTPUT_DIR}/S4_22_{STATION}_rank_modelblock.csv')

# %% [markdown]
# ## Step 9 · Combinations, and how to diagnose them
#
# **Question 5.** With two candidates the exhaustive search is three subsets, so
# there is no room for a stepwise procedure and none is used. The diagnostics
# that follow are the substance of the question.

# %%
vif = ip.vif_table(df, DRIVERS)
display(vif)
vif.to_csv(f'{OUTPUT_DIR}/S4_20_{STATION}_vif.csv')

# %%
subsets = ip.subset_search(
    df, TARGET, DRIVERS, OPERATORS, dt_hours=SAMPLING_HOURS,
    n_splits=CV_SPLITS, min_train_frac=MIN_TRAIN_FRAC, alpha=RIDGE_ALPHA,
    full_record=df)
display(subsets)
subsets.to_csv(f'{OUTPUT_DIR}/S4_23_{STATION}_subsets.csv')

# %%
ip.plot_subset_search(
    subsets, title='Exhaustive subset search, legacy drivers',
    save_path=OUTPUT_DIR, filename=f'S4_F14_{STATION}_subsets')
plt.show()

# %% [markdown]
# ### Which member is doing the work
#
# Two diagnostics, deliberately different. **Leave-one-out gain** removes a
# channel and refits: what would it cost to build the system without it?
# **Permutation importance** shuffles a channel in the test block with the model
# held fixed: how much does the model as built rely on it? They disagree exactly
# when collinearity lets a refit recover the lost information, and that
# disagreement is the diagnosis.

# %%
best_subset = subsets.index[0].split(' + ')
print(f'Best subset by MAE: {" + ".join(best_subset)}')

gain = ip.incremental_gain(
    df, TARGET, best_subset, OPERATORS, dt_hours=SAMPLING_HOURS,
    n_splits=CV_SPLITS, min_train_frac=MIN_TRAIN_FRAC, alpha=RIDGE_ALPHA)
importance = ip.permutation_importance(
    df, TARGET, best_subset, OPERATORS, dt_hours=SAMPLING_HOURS,
    n_splits=CV_SPLITS, min_train_frac=MIN_TRAIN_FRAC, alpha=RIDGE_ALPHA,
    n_repeats=N_REPEATS)
display(gain)
display(importance)
gain.to_csv(f'{OUTPUT_DIR}/S4_24_{STATION}_gain.csv')
importance.to_csv(f'{OUTPUT_DIR}/S4_25_{STATION}_importance.csv')

# %%
ip.plot_diagnostics(
    gain, importance,
    title=f'Contribution of each channel in "{" + ".join(best_subset)}"',
    save_path=OUTPUT_DIR, filename=f'S4_F15_{STATION}_diagnostics')
plt.show()

# %% [markdown]
# ### Out-of-period check
#
# The models are fitted on the unbroken block and applied to a later block,
# without refitting. This is the hardest test available here: a different part of
# the record, separated by a 271-day outage.

# %%
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

holdout_rows = []
for label, cols in [('best subset', best_subset),
                    ('best single', [rank.index[0]])]:
    Xtr = ip.build_features(model_df, cols, OPERATORS, dt_hours=SAMPLING_HOURS)
    tr = Xtr.join(model_df[TARGET].rename('__y__')).dropna()
    Xte = ip.build_features(holdout_df, cols, OPERATORS,
                            dt_hours=SAMPLING_HOURS)
    te = Xte.join(holdout_df[TARGET].rename('__y__')).dropna()
    if len(tr) < 200 or len(te) < 100:
        print(f'  {label}: too few rows, skipped')
        continue
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(tr.drop(columns='__y__'), tr['__y__'])
    err = model.predict(te.drop(columns='__y__')) - te['__y__'].values
    holdout_rows.append({
        'model': f'{label}: {" + ".join(cols)}',
        'n_test': len(te),
        'MAE_mdeg': round(float(np.abs(err).mean()), 4),
        'RMSE_mdeg': round(float(np.sqrt((err ** 2).mean())), 4),
    })
holdout = pd.DataFrame(holdout_rows).set_index('model')
display(holdout)
holdout.to_csv(f'{OUTPUT_DIR}/S4_26_{STATION}_holdout.csv')

# %% [markdown]
# ## Step 10 · The diagnostic protocol
#
# Question 5 asks not only which combination wins but how to tell. The procedure
# used above is the answer, and it transfers to any station and any sensor set.
# It is the sibling study's protocol with one rule added — rule 2 — which this
# era's spectral work forced.

# %%
protocol = pd.DataFrame({
    'step': range(1, 11),
    'rule': [
        'bound the operator scan by the mechanism and by half the dominant period',
        'estimate the spectrum before imputing anything, and control for the '
        'sampling pattern',
        'fit each driver operator on the band of interest, not the raw series',
        'score without autoregression',
        'rolling-origin folds, report the spread',
        'exhaustive subset search when the candidate set is small',
        'gain and permutation importance, never coefficients under high VIF',
        'weight by availability over the whole record',
        'carry a negative control',
        'confirm on an out-of-period block',
    ],
    'failure it prevents': [
        'a physically impossible optimum, or an aliased lead read as a delay',
        'letting the filling method decide which frequencies exist, and reading '
        'an outage pattern as a structural cycle',
        'a boundary optimum reported as a physical time constant',
        'every driver tying behind the autoregressive term',
        'a ranking that depends on where the single split fell',
        'selection bias and instability under collinearity',
        'importance attributed by collinearity rather than by content',
        'choosing a channel that is absent when it is needed',
        'mistaking shared diurnal shape for predictive content',
        'a combination that fits one period and transfers to none',
    ],
}).set_index('step')
display(protocol)
protocol.to_csv(f'{OUTPUT_DIR}/S4_27_{STATION}_protocol.csv')

# %% [markdown]
# ## Step 11 · The error budget
#
# **Question 6, first half.** A single forecast error says how well the system
# does. It does not say what the error is made of, and therefore does not say
# what would improve it. The ladder adds one component at a time and scores every
# rung at every horizon.
#
# **No rung carries a linear trend.** Extrapolating a straight line across this
# record is the single most damaging term available, and leaving it inside the
# ladder would make every gain below it a measure of recovery from that one
# decision rather than of the component's own worth. It is priced on its own
# immediately after the ladder.

# %%
budget = lp.error_budget(
    df, TARGET, DRIVERS, HORIZONS_LADDER, origins=LADDER_ORIGINS,
    n_lags=N_LAGS, periods=lp.HARMONIC_PERIODS, alpha=RIDGE_ALPHA)
display(budget)
budget.to_csv(f'{OUTPUT_DIR}/S4_28_{STATION}_error_budget.csv')
print(f'scored samples per horizon: {budget.attrs["n"]}')

# %%
gains = lp.budget_gains(budget)
display(gains)
gains.to_csv(f'{OUTPUT_DIR}/S4_29_{STATION}_budget_gains.csv')

# %%
lp.plot_error_budget(
    budget, gains=gains,
    title='Error budget: what each component buys, by horizon',
    save_path=OUTPUT_DIR, filename=f'S4_F16_{STATION}_error_budget')
plt.show()

# %% [markdown]
# ### What the trend term costs

# %%
trend_probe = lp.error_budget(
    df, TARGET, DRIVERS, HORIZONS_LADDER, origins=LADDER_ORIGINS,
    n_lags=N_LAGS, periods=lp.HARMONIC_PERIODS, alpha=RIDGE_ALPHA,
    ladder=lp.TREND_PROBE)
trend_probe.loc['cost of the trend'] = (trend_probe.loc['seasonal + trend']
                                        - trend_probe.loc['seasonal, no trend'])
display(trend_probe.round(3))
trend_probe.to_csv(f'{OUTPUT_DIR}/S4_30_{STATION}_trend_cost.csv')

# %% [markdown]
# ### The same ladder with the drivers known in advance

# %%
budget_future = lp.error_budget(
    df, TARGET, DRIVERS, HORIZONS_LADDER, origins=LADDER_ORIGINS,
    n_lags=N_LAGS, periods=lp.HARMONIC_PERIODS, alpha=RIDGE_ALPHA,
    future_regressors=True)
comparison = pd.DataFrame({
    'lagged': budget.loc['+ drivers'],
    'future': budget_future.loc['+ drivers'],
})
comparison['gain_%'] = (100 * (comparison['lagged'] - comparison['future'])
                        / comparison['lagged']).round(2)
display(comparison)
comparison.to_csv(f'{OUTPUT_DIR}/S4_31_{STATION}_regime_gap.csv')

# %% [markdown]
# ## Step 12 · NeuralProphet at the short horizons
#
# **Question 6, second half.** NeuralProphet cannot span a year: `n_forecasts`
# would need 8760 output heads. It is run on the longest unbroken block at
# horizons up to one week.
#
# **The annual term is off here, and this cell is the record of that.** An
# earlier version of this text claimed the term was enabled; it was not.
# `ip.run_horizon_experiment` fixed `yearly_seasonality=False`, so the
# configuration below carries a daily seasonality, an autoregressive term and the
# drivers, and no annual component at all. The parameter is now explicit rather
# than fixed, and Step 13b enables it on a record long enough to identify it.
#
# Two regressor regimes, exactly as in the sibling study: **lagged**, which is
# what a deployed system has, and **future**, which is what the same system
# coupled to a perfect weather forecast would have.

# %%
np_results, np_forecasts = {}, {}
if RUN_NEURALPROPHET:
    feats = ip.build_features(model_df, DRIVERS, OPERATORS,
                              dt_hours=SAMPLING_HOURS)
    frame = feats.join(model_df[TARGET])
    for regime in ('lagged', 'future'):
        res, fc = ip.run_horizon_experiment(
            frame, TARGET, DRIVERS, max(HORIZONS_NP), HORIZONS_NP,
            freq=ANALYSIS_FREQ, regime=regime, n_lags=N_LAGS,
            epochs=NP_EPOCHS, quantiles=QUANTILES,
            train_frac=NP_TRAIN_FRAC, keep_forecast=True)
        base = ip.baseline_errors(model_df[TARGET], HORIZONS_NP, fc.index,
                                  dt_hours=SAMPLING_HOURS)
        label = f'model block, {regime}'
        np_results[label] = ip.add_skill(res, base)
        np_forecasts[label] = fc
        print(f'{label}: scored {list(res.index)}')

# %%
if np_results:
    for label, res in np_results.items():
        print(f'\n=== {label} ===')
        display(res)
        tag = label.replace(', ', '_').replace(' ', '')
        res.to_csv(f'{OUTPUT_DIR}/S4_32_{STATION}_np_{tag}.csv')

# %%
if np_results:
    ip.plot_horizon_skill(
        np_results, title='NeuralProphet on the longest unbroken block',
        save_path=OUTPUT_DIR, filename=f'S4_F17_{STATION}_np_skill')
    plt.show()

# %%
if np_results:
    ip.plot_uncertainty(
        np_results, title='Calibration of the 90 % prediction interval',
        save_path=OUTPUT_DIR, filename=f'S4_F18_{STATION}_np_uncertainty')
    plt.show()

# %%
if np_results:
    limits = pd.DataFrame([
        {'configuration': label,
         'last horizon beating persistence [h]':
             ip.horizon_limit(res, 'skill_vs_persistence'),
         'last horizon beating seasonal naive [h]':
             ip.horizon_limit(res, 'skill_vs_seasonal_naive'),
         'MAE at 24 h [mdeg]': res['MAE_mdeg'].get(24, np.nan),
         'coverage at 24 h [%]': (res['coverage_%'].get(24, np.nan)
                                  if 'coverage_%' in res else np.nan)}
        for label, res in np_results.items()
    ]).set_index('configuration')
    display(limits)
    limits.to_csv(f'{OUTPUT_DIR}/S4_33_{STATION}_horizon_limits.csv')

# %% [markdown]
# ### One stretch of the forecast, drawn

# %%
if np_forecasts:
    label = 'model block, lagged'
    ip.plot_fan(
        np_forecasts[label], model_df[TARGET], horizon=24,
        quantiles=QUANTILES, days=10,
        title=f'{label} — 24 hours ahead, with the 90 % interval',
        save_path=OUTPUT_DIR, filename=f'S4_F19_{STATION}_fan')
    plt.show()

# %% [markdown]
# ### What the joint multi-horizon head costs
#
# One fit produced every horizon, which is efficient but not free: the network
# shares one set of weights across 168 output steps. Refitting with
# `n_forecasts` set to exactly one horizon isolates the cost.

# %%
if RUN_NEURALPROPHET:
    dedicated_rows = []
    for h in (1, 24):
        res = ip.run_horizon_experiment(
            frame, TARGET, DRIVERS, h, [h], freq=ANALYSIS_FREQ,
            regime='lagged', n_lags=N_LAGS, epochs=NP_EPOCHS,
            quantiles=QUANTILES, train_frac=NP_TRAIN_FRAC)
        joint = np_results.get('model block, lagged')
        dedicated_rows.append({
            'horizon_h': h,
            'MAE_dedicated': res['MAE_mdeg'].iloc[0] if len(res) else np.nan,
            'MAE_joint_head': (joint['MAE_mdeg'].get(h, np.nan)
                               if joint is not None else np.nan),
        })
    dedicated = pd.DataFrame(dedicated_rows).set_index('horizon_h')
    dedicated['joint_head_penalty_%'] = (
        100 * (dedicated['MAE_joint_head'] - dedicated['MAE_dedicated'])
        / dedicated['MAE_dedicated']).round(2)
    display(dedicated)
    dedicated.to_csv(f'{OUTPUT_DIR}/S4_34_{STATION}_dedicated_vs_joint.csv')

# %% [markdown]
# ## Step 13 · Sensitivity
#
# Every result above rests on decisions that could defensibly have gone another
# way. The headline is the autoregressive rung's error one day ahead.

# %%
HEADLINE_H = 24


def headline(frame=None, drivers=None, periods=None, n_lags=N_LAGS,
             origins=LADDER_ORIGINS, coeff=None, rung='+ AR'):
    """Return the ladder's error for one rung at the headline horizon."""
    data = df if frame is None else frame
    if coeff is not None:
        data = data.copy()
        data[TARGET] = tc.compensate(data, temp_col='tair', coeff=coeff,
                                     normalise=True)
    tab = lp.error_budget(
        data, TARGET, DRIVERS if drivers is None else drivers,
        [HEADLINE_H], origins=origins, n_lags=n_lags,
        periods=lp.HARMONIC_PERIODS if periods is None else periods,
        alpha=RIDGE_ALPHA)
    return tab.loc[rung, HEADLINE_H]


variations = {
    'baseline': {},
    'compensation coefficient 0.00163 (fitted)': {'coeff': 0.00163},
    'compensation coefficient 0.010': {'coeff': 0.010},
    'no semi-annual term': {'periods': {k: v for k, v
                                        in lp.HARMONIC_PERIODS.items()
                                        if k != 'semiannual'}},
    'no seasonal terms at all': {'periods': {}},
    'autoregressive depth 12': {'n_lags': 12},
    'autoregressive depth 48': {'n_lags': 48},
    'model block only': {'frame': model_df},
    'drivers dropped': {'drivers': []},
    'origins 0.4/0.5/0.6': {'origins': (0.4, 0.5, 0.6)},
}
sens = lp.sensitivity_sweep(headline, variations)
display(sens)
sens.to_csv(f'{OUTPUT_DIR}/S4_35_{STATION}_sensitivity.csv')

# %%
lp.plot_sensitivity(
    sens, title=f'Sensitivity of the {HEADLINE_H} h autoregressive error',
    save_path=OUTPUT_DIR, filename=f'S4_F20_{STATION}_sensitivity')
plt.show()

# %% [markdown]
# ## Step 13b · How much of the record NeuralProphet can actually use
#
# Step 2 restricted the autoregressive work to the longest unbroken block on the
# grounds that an autoregressive model does not tolerate gaps. That premise is
# too strong, and it has been costing this study most of its record.
#
# NeuralProphet needs *windows*, not a continuous series. With `n_lags = 0` the
# trend and the Fourier terms are functions of the timestamp alone, so gaps are
# irrelevant and the whole era is available already. With `n_lags > 0` the model
# needs `n_lags` consecutive observations followed by `n_forecasts` more, and
# those windows exist in every block, not only in the longest one.
#
# That reframes what imputation is for. Filling a hole is not a way of making the
# record continuous; it is a way of buying back the windows the hole destroyed.
# One missing hour costs a full window's worth of samples however short it is, so
# the many brief interruptions are cheap to fill and expensive to leave, while a
# months-long outage is the reverse. This step measures that trade instead of
# assuming which way it falls, and then compares three regimes against Step 12 as
# the control:
#
# | | span | fabricated | role |
# |---|---|---|---|
# | control | `MODEL_WINDOW`, 617 d | as Step 12 | the existing result |
# | **A** | full era, no fill | none | decomposition, `n_lags = 0` |
# | **B** | full era, holes to 6 h filled | ~1.5 % | prediction, segmented |
# | **C** | one bridged block | ~14 % | the only continuous series available |
#
# Nothing in Step 12 is re-run or altered. Every number this step produces is new.

# %% [markdown]
# ### What each gap tolerance buys
#
# Step 2 swept the tolerance to three days and found no block holding two annual
# cycles. The sweep is extended here, because the question is not whether a
# tolerant block exists but what admitting one costs.

# %%
sweep_rows = []
for tol in (6, 24, 72, 168, 360, 720, 1080, 1440, 2160):
    wr = lp.window_report(df, [TARGET] + DRIVERS, lp.YEAR_DAYS,
                          min_cycles=2.0, min_days=60, max_gap_hours=tol)
    if not len(wr):
        continue
    best = wr.iloc[0]
    sweep_rows.append({
        'tolerance_h': tol,
        'tolerance_d': round(tol / 24, 1),
        'n_blocks': len(wr),
        'longest_days': round(float(best['days']), 1),
        'annual_cycles': float(best['cycles']),
        'coverage_%': float(best['coverage_%']),
        'fabricated_%': round(100.0 - float(best['coverage_%']), 1),
        'qualifies': bool(best['qualifies']),
        'start': str(pd.Timestamp(best['start']).date()),
        'end': str(pd.Timestamp(best['end']).date()),
    })
block_sweep = pd.DataFrame(sweep_rows).set_index('tolerance_h')
display(block_sweep)
block_sweep.to_csv(f'{OUTPUT_DIR}/S4_37_{STATION}_block_sweep.csv')

qualifying = block_sweep[block_sweep['qualifies']]
if len(qualifying):
    first = qualifying.iloc[0]
    BRIDGE_TOLERANCE_H = int(qualifying.index[0])
    print(f'The two-cycle rule is first met at a tolerance of '
          f'{BRIDGE_TOLERANCE_H} h ({BRIDGE_TOLERANCE_H / 24:.0f} days): '
          f'{first["longest_days"]:.0f} d = {first["annual_cycles"]:.2f} cycles, '
          f'{first["fabricated_%"]:.1f} % of it fabricated.')
else:
    BRIDGE_TOLERANCE_H = None
    print('No tolerance produces a block of two annual cycles.')

# %% [markdown]
# ### Windows gained against hours invented
#
# The tolerance sweep counts days. What a model consumes is windows, and the two
# do not rank the policies the same way.

# %%
window_policies_table = lp.window_policies(
    df[TARGET],
    {'control: model block': (model_df[TARGET], None),
     'full era, no fill': 0,
     'full era, fill 6 h': 6,
     'full era, fill 24 h': 24,
     'full era, fill 72 h': 72},
    n_lags=N_LAGS, n_forecasts=max(HORIZONS_NP),
    dt_hours=SAMPLING_HOURS, baseline='control: model block')
display(window_policies_table[
    ['span_days', 'annual_cycles', 'n_windows', 'hours_observed',
     'hours_fabricated', 'fabricated_%', 'windows_vs_baseline',
     'windows_per_hour_filled']])
window_policies_table.to_csv(f'{OUTPUT_DIR}/S4_38_{STATION}_window_policies.csv')

# %%
lp.plot_window_accounting(
    window_policies_table,
    title='What each fill policy buys, against what it invents',
    save_path=OUTPUT_DIR, filename=f'S4_F21_{STATION}_window_accounting')
plt.show()

# %% [markdown]
# ### Is the record one series or two?
#
# The 438-day outage is long enough for the instrument to have been serviced,
# remounted or replaced, and the absolute level of an inclinometer carries no
# meaning across such an event. A model fitted straight through it would read any
# offset as structural drift, so the offset has to be looked for first.
#
# The naive comparison confounds the offset with the season, because the last
# observations before the outage and the first after it sit at different points
# of the annual cycle. Air temperature, which the instrument does not influence,
# shows how much of the apparent step is calendar; subtracting the fitted
# harmonic model removes that contribution directly.

# %%
left_block = df.loc[:MODEL_WINDOW[1], TARGET].dropna()
right_block = df.loc[HOLDOUT[0]:, TARGET].dropna()
step_test = lp.baseline_step_test(
    df[TARGET], left_block.index.max(), right_block.index.min(),
    window_days=30, control=df['tair'], deseason=harm_fit)
display(step_test)
step_test.to_csv(f'{OUTPUT_DIR}/S4_39_{STATION}_baseline_step.csv')

raw_step = step_test.loc['response', 'step']
net_step = step_test.loc['response, deseasoned', 'step']
print(f'Interruption: {step_test.attrs["interruption_days"]:.0f} days.')
print(f'Raw step {raw_step:+.1f} mdeg; once the seasonal expectation is removed, '
      f'{net_step:+.1f} mdeg — {100 * (1 - abs(net_step / raw_step)):.0f} % of '
      f'the apparent step was the calendar.')

# %% [markdown]
# ### Regime A — decomposition, control against the whole era
#
# This is the test the framework in the companion manuscript asserts without
# evidence: that a window of one to two annual cycles suffices to fit a yearly
# Fourier component. The same model is fitted on 1.69 cycles and on 6.57, and
# both are measured against the harmonic fit of Step 5, whose single-basis design
# gives an annual signal exactly one place to go.

# %%
decomp_rows, decomp_components = [], {}
for label, sub in [('control: model block', model_df), ('full era', df)]:
    ref_table, _, _ = lp.fit_harmonics(sub[TARGET], lp.HARMONIC_PERIODS,
                                       trend=False)
    comps, _ = lc.decompose_series(sub[TARGET], freq=ANALYSIS_FREQ,
                                   epochs=NP_EPOCHS, yearly=True,
                                   weekly=False, daily=True)
    decomp_components[label] = comps
    decomp_rows.append(lp.compare_decompositions(
        comps, harmonic=ref_table, label=label, t0=sub.index.min()))

decomposition = pd.concat(decomp_rows)
display(decomposition[['std_trend', 'std_season_yearly', 'trend_share_%',
                       'np_annual_amplitude', 'ref_annual_amplitude',
                       'amplitude_ratio', 'phase_gap_days', 'fit_MAE_mdeg']])
decomposition.to_csv(f'{OUTPUT_DIR}/S4_40_{STATION}_decomposition.csv')

# %%
lp.plot_decomposition(
    decomp_components['full era'], harmonic_fit=harm_fit,
    title='NeuralProphet decomposition of the whole era, against the harmonic fit',
    save_path=OUTPUT_DIR, filename=f'S4_F22_{STATION}_decomposition')
plt.show()

# %% [markdown]
# ### Regime B — prediction across every complete stretch
#
# NeuralProphet's own `drop_missing` is the obvious way to fit an autoregressive
# model on a gapped record, and in version 0.8.0 it does not work: training on a
# frame that actually requires a drop returns a NaN loss from the first epoch and
# leaves every weight NaN, and the prediction path raises a length mismatch. Both
# were verified on this record.
#
# The route that does work is the library's multi-series support. Interruptions
# up to the interpolation limit of Step 6 are filled, every stretch that survives
# is presented as its own series, and one set of shared components is fitted
# across all of them. The seasonal terms therefore see all six and a half years
# while no autoregressive window is ever formed across an outage, and — because
# normalisation is per series — an offset introduced during one of them cannot
# propagate into the fit as structural movement.

# %%
segment_results, segment_forecasts = {}, {}
if RUN_NEURALPROPHET:
    full_frame = ip.build_features(df, DRIVERS, OPERATORS,
                                   dt_hours=SAMPLING_HOURS).join(df[TARGET])
    for regime in ('lagged', 'future'):
        res, fc = ip.run_horizon_segments(
            full_frame, TARGET, DRIVERS, max(HORIZONS_NP), HORIZONS_NP,
            freq=ANALYSIS_FREQ, regime=regime, n_lags=N_LAGS,
            epochs=NP_EPOCHS, quantiles=QUANTILES, train_frac=NP_TRAIN_FRAC,
            yearly=True, fill_limit_h=SEGMENT_FILL_H, n_changepoints=0,
            keep_forecast=True)
        base = ip.baseline_errors(df[TARGET], HORIZONS_NP, fc.index,
                                  dt_hours=SAMPLING_HOURS)
        label = f'full era, {regime}'
        segment_results[label] = ip.add_skill(res, base)
        segment_forecasts[label] = fc
        print(f'{label}: {res.attrs["n_segments"]} stretches, '
              f'{res.attrs["train_rows"]} training rows, '
              f'{res.attrs["hours_fabricated"]:.0f} h fabricated')

# %%
if segment_results:
    for label, res in segment_results.items():
        print(f'\n=== {label} ===')
        display(res)
        tag = label.replace(', ', '_').replace(' ', '')
        res.to_csv(f'{OUTPUT_DIR}/S4_41_{STATION}_np_{tag}.csv')

# %% [markdown]
# The comparison that matters is against Step 12, which fitted the same model on
# the longest unbroken block alone.

# %%
if segment_results and np_results:
    compare_rows = []
    for regime in ('lagged', 'future'):
        ctrl = np_results.get(f'model block, {regime}')
        seg = segment_results.get(f'full era, {regime}')
        if ctrl is None or seg is None:
            continue
        for h in HORIZONS_NP:
            if h not in ctrl.index or h not in seg.index:
                continue
            compare_rows.append({
                'regime': regime, 'horizon_h': h,
                'MAE_control': ctrl['MAE_mdeg'][h],
                'MAE_full_era': seg['MAE_mdeg'][h],
                'change_%': round(100 * (seg['MAE_mdeg'][h]
                                         - ctrl['MAE_mdeg'][h])
                                  / ctrl['MAE_mdeg'][h], 1),
                'coverage_control': ctrl.get('coverage_%', pd.Series()).get(h),
                'coverage_full_era': seg.get('coverage_%', pd.Series()).get(h),
                'width_control': ctrl.get('PI_width_mdeg', pd.Series()).get(h),
                'width_full_era': seg.get('PI_width_mdeg', pd.Series()).get(h),
            })
    window_gain = pd.DataFrame(compare_rows).set_index(['regime', 'horizon_h'])
    display(window_gain)
    window_gain.to_csv(f'{OUTPUT_DIR}/S4_42_{STATION}_window_gain.csv')

# %%
if segment_results:
    ip.plot_horizon_skill(
        segment_results,
        title='NeuralProphet across every complete stretch of the era',
        save_path=OUTPUT_DIR, filename=f'S4_F23_{STATION}_segment_skill')
    plt.show()

# %% [markdown]
# ### Regime C — the bridged block, and what the fill costs it
#
# The only way to obtain a genuinely continuous series of two annual cycles is to
# invent the interruptions inside it. That is done here for completeness and is
# not recommended: the holes bridged are far longer than anything the Step 6
# benchmark licenses. The bias it introduces is measured rather than asserted, by
# fitting the harmonic model twice — once on the bridged series, once with the
# fabricated positions masked out again.

# %%
bridged_bias = None
if BRIDGE_TOLERANCE_H is not None:
    bridged, fabricated, bridge_summary = lp.bridge_blocks(
        df, [TARGET] + DRIVERS, BRIDGE_TOLERANCE_H, policy=policy,
        dt_hours=SAMPLING_HOURS)
    print({k: str(v) for k, v in bridge_summary.items()})

    tab_bridged, _, _ = lp.fit_harmonics(bridged[TARGET], lp.HARMONIC_PERIODS,
                                         trend=False)
    tab_observed, _, _ = lp.fit_harmonics(bridged[TARGET].mask(fabricated),
                                          lp.HARMONIC_PERIODS, trend=False)
    bridged_bias = pd.DataFrame({
        'bridged': tab_bridged['amplitude_mdeg'],
        'observed_only': tab_observed['amplitude_mdeg'],
    })
    bridged_bias['bias_%'] = (100 * (bridged_bias['bridged']
                                     / bridged_bias['observed_only'] - 1)).round(2)
    for key, value in bridge_summary.items():
        bridged_bias.attrs[key] = value
    display(bridged_bias)
    bridged_bias.to_csv(f'{OUTPUT_DIR}/S4_43_{STATION}_bridged_bias.csv')
    print(f'Bridging {bridge_summary["fabricated_%"]:.1f} % of the block biases '
          f'the annual amplitude by '
          f'{bridged_bias.loc["annual", "bias_%"]:+.1f} %.')

# %% [markdown]
# ### The guard
#
# The whole argument rests on nothing beyond the licensed limit being invented.
# That is asserted nowhere and checked here.

# %%
_grid = pd.date_range(df.index.min(), df.index.max(), freq=ANALYSIS_FREQ)
_filled = ip._fill_short_runs(df[TARGET].reindex(_grid), SEGMENT_FILL_H)
_missing = _filled.isna().to_numpy()
_edges = np.flatnonzero(np.diff(np.r_[0, _missing.astype(int), 0]))
_runs = (_edges[1::2] - _edges[::2]) * SAMPLING_HOURS
assert _runs.min() > SEGMENT_FILL_H, (
    f'a run of {_runs.min()} h survived a {SEGMENT_FILL_H} h fill limit')
_invented = int(df[TARGET].isna().sum() - _filled.isna().sum())
assert _invented == window_policies_table.loc['full era, fill 6 h',
                                              'hours_fabricated'], \
    'the fill guard and the window accounting disagree'
print(f'Regime B invents {_invented} h. Every surviving hole is longer than '
      f'{SEGMENT_FILL_H} h; the shortest is {_runs.min():.0f} h and the longest '
      f'{_runs.max():.0f} h. The {step_test.attrs["interruption_days"]:.0f}-day '
      f'outage is untouched.')

# %% [markdown]
# ## Step 14 · Verdicts

# %%
best_low = peaks_low.iloc[0]
semi = peaks_low[(peaks_low['period_days'] > 170)
                 & (peaks_low['period_days'] < 195)]
ann = sliding[sliding['component'] == 'annual']['amplitude_mdeg']
verdict_rows = [
    {'question': '1 · periodicities present',
     'answer': f'strongest at {best_low["label"]} (power '
               f'{best_low["power"]:.3f}, window {best_low["window_power"]:.3f}); '
               f'semi-annual '
               f'{"confirmed at " + semi.iloc[0]["label"] if len(semi) else "absent"}; '
               f'diurnal at {peaks_high.iloc[0]["label"]}'},
    {'question': '1 · spurious peaks caught',
     'answer': 'peaks whose window-function power exceeds their signal power '
               'are outage artefacts; see the peak table'},
    {'question': '2 · is the semi-annual independent?',
     'answer': f'phase-locked to the annual term (circular sd '
               f'{lock["circ_sd_deg"]} deg against {lock["null_sd_deg"]} deg '
               f'for an unlocked null) -> second harmonic, not a separate '
               f'180 d process'},
    {'question': '2 · are the components stable?',
     'answer': f'annual amplitude ranges {ann.min():.1f}-{ann.max():.1f} mdeg '
               f'across {SLIDE_WINDOW_DAYS} d windows'},
    {'question': '3 · imputation limit',
     'answer': '; '.join(f'{m}: {int(r["max_gap_h"])} h'
                         for m, r in policy.iterrows())},
    {'question': '4 · best single predictor',
     'answer': f'{rank.index[0]} (R2 {rank["R2"].iloc[0]:.3f}); control '
               f'{CONTROL} ranks {list(rank.index).index(CONTROL) + 1} of '
               f'{len(rank)}'},
    {'question': '4 · signed-delay control',
     'answer': f'no forcing gains more than {LEAD_TOLERANCE_R2} R2 from a '
               f'lead: {controls_ok}'
               + (f'; largest lead {leads["delay_h"].min():+.0f} h buying '
                  f'{leads["r2_lost_to_causality"].max():+.4f} R2'
                  if len(leads) else '')},
    {'question': '5 · best combination',
     'answer': f'{subsets.index[0]} (MAE {subsets["MAE_mdeg"].iloc[0]:.2f} '
               f'mdeg, {subsets["MAE_vs_best_%"].iloc[1]:.1f} % better than '
               f'the runner-up); utility winner '
               f'{subsets["utility"].idxmax()}'},
    {'question': '5 · out-of-period',
     'answer': ('; '.join(f'{m}: {r["MAE_mdeg"]:.2f} mdeg'
                          for m, r in holdout.iterrows())
                if len(holdout) else 'not evaluable')},
    {'question': '6 · error budget, 24 h',
     'answer': '; '.join(f'{k} {v:.2f}' for k, v
                         in budget[HEADLINE_H].items())},
    {'question': '6 · the trend term',
     'answer': f'extrapolating it costs '
               f'{trend_probe.loc["cost of the trend"].min():.0f}-'
               f'{trend_probe.loc["cost of the trend"].max():.0f} mdeg at '
               f'every horizon'},
]
if np_results:
    for label, res in np_results.items():
        cov = (res['coverage_%'].get(24, float('nan'))
               if 'coverage_%' in res else float('nan'))
        verdict_rows.append({
            'question': f'6 · {label}',
            'answer': f'beats persistence out to '
                      f'{ip.horizon_limit(res, "skill_vs_persistence")} h; '
                      f'coverage at 24 h {cov:.0f} % against a nominal '
                      f'{res.attrs.get("nominal_coverage_%", 90):.0f} %'})

_full_era_row = window_policies_table.loc['full era, fill 6 h']
verdict_rows.append({
    'question': '7 · usable record',
    'answer': f'the longest block gives '
              f'{window_policies_table.loc["control: model block", "annual_cycles"]:.2f} '
              f'annual cycles; fitting across every complete stretch gives '
              f'{_full_era_row["annual_cycles"]:.2f} for '
              f'{_full_era_row["fabricated_%"]:.1f} % fabricated hours, and '
              f'{_full_era_row["windows_vs_baseline"]:.2f} times the windows'})
verdict_rows.append({
    'question': '7 · one series or two?',
    'answer': f'the {step_test.attrs["interruption_days"]:.0f}-day outage shows a '
              f'{step_test.loc["response", "step"]:+.1f} mdeg step, of which '
              f'{step_test.loc["response, deseasoned", "step"]:+.1f} mdeg '
              f'survives removing the seasonal expectation'})
if not decomposition.empty:
    verdict_rows.append({
        'question': '7 · does a longer window help the decomposition?',
        'answer': '; '.join(
            f'{label}: annual amplitude '
            f'{row["amplitude_ratio"]:.2f} x the harmonic reference, trend holds '
            f'{row["trend_share_%"]:.0f} % of the low-frequency variance'
            for label, row in decomposition.iterrows())})
if segment_results and np_results:
    for regime in ('lagged', 'future'):
        ctrl = np_results.get(f'model block, {regime}')
        seg = segment_results.get(f'full era, {regime}')
        if ctrl is None or seg is None:
            continue
        verdict_rows.append({
            'question': f'7 · full era against the block, {regime}',
            'answer': f'MAE at 24 h {ctrl["MAE_mdeg"].get(24, float("nan")):.2f} '
                      f'-> {seg["MAE_mdeg"].get(24, float("nan")):.2f} mdeg; '
                      f'interval width '
                      f'{ctrl["PI_width_mdeg"].get(24, float("nan")):.0f} -> '
                      f'{seg["PI_width_mdeg"].get(24, float("nan")):.0f} mdeg at '
                      f'coverage {ctrl["coverage_%"].get(24, float("nan")):.0f} '
                      f'-> {seg["coverage_%"].get(24, float("nan")):.0f} %'})
if bridged_bias is not None:
    verdict_rows.append({
        'question': '7 · the bridged block',
        'answer': f'{bridged_bias.attrs["span_days"]:.0f} d = '
                  f'{bridged_bias.attrs["annual_cycles"]:.2f} cycles at '
                  f'{bridged_bias.attrs["fabricated_%"]:.1f} % fabricated, '
                  f'{bridged_bias.attrs["holes_beyond_policy"]} holes beyond what '
                  f'the benchmark licenses; biases the annual amplitude by '
                  f'{bridged_bias.loc["annual", "bias_%"]:+.1f} %'})
verdicts = pd.DataFrame(verdict_rows).set_index('question')
display(verdicts)
verdicts.to_csv(f'{OUTPUT_DIR}/S4_36_{STATION}_verdicts.csv')

# %% [markdown]
# ---
#
# The full written evaluation is in
# `report/inclination_prediction_legacy_report.pdf`.
