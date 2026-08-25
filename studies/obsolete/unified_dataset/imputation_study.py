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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Study · What the record is made of, and what imputation can do about it
#
# This study consumes the unified dataset built by `build_unified_dataset.py`
# and never reads the raw archive. It asks one question in four parts:
#
# 1. **What is actually there** — per station, per era, per channel: how much
#    record exists, when, and how complete it is.
# 2. **What is missing, and why** — the anatomy of the gaps, their length, their
#    season, and the mechanism that produced them. The mechanism decides what is
#    reconstructible; the length decides by what.
# 3. **What can be filled, by what method, to what accuracy** — six fillers
#    measured against gaps of known length, each carrying a distribution-free
#    interval rather than a point estimate alone.
# 4. **How long a continuous stretch that buys** — the trade between record
#    length gained and data invented, and the recommendation that follows.
#
# ## The three findings that shape everything below
#
# **Every channel fails at once.** The missingness is not random and not
# conditional on an observed covariate: when the inclinometer is missing, the
# air temperature, the humidity and the battery are missing too, on essentially
# every gap. That is outage-driven missingness in its purest form, and it has a
# hard consequence — *no covariate-conditioned method can ever fill a real gap
# in this record*, because the covariates are gone with it.
#
# **The benchmark cannot see that.** An injected gap has its drivers intact, so
# a driver-based filler scores well on the benchmark and would fail completely
# in production. The study measures it anyway and reports the discrepancy,
# because the discrepancy is the point.
#
# **The two eras are two instruments.** Any stretch crossing 2025-02-21 rests on
# an estimated offset, and the diurnal-amplitude ratio at the changeover says
# how much that estimate can be trusted.

# %% [markdown]
# ## Imports and configuration

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../..'))

import ud_lib as ud

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 80)

# %% [markdown]
# ### Parameters
#
# | Parameter | Purpose |
# |---|---|
# | `DATA_DIR` | Where `build_unified_dataset.py` wrote its output. |
# | `TARGET` | Column the imputation work operates on. |
# | `GAP_LENGTHS` | Gap lengths the fillers are benchmarked against, in hours. |
# | `N_TRIALS` | Injection sites per gap length. |
# | `TOLERANCES` | Accepted mean absolute errors, each generating one policy. |
# | `CONFORMAL_LEVEL` | Nominal coverage of the calibrated interval. |

# %%
DATA_DIR = '../../../data/interim/unified'
OUTPUT_DIR = 'outputs'
STATION = ud.TARGET_STATION
STATIONS = ud.STATIONS

TARGET = 'inc_comp_joined'     # the continuous-across-eras compensated series
TARGET_PER_ERA = 'inc_comp'    # the per-era-anchored one, for era-local work
DRIVERS = ['tair', 'rh']

GAP_LENGTHS = (1, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1080)
N_TRIALS = 30
FAST_METHODS = ('interpolate', 'seasonal_naive', 'kalman', 'harmonic',
                'drivers')
TOLERANCES = (2.0, 5.0, 10.0, 15.0, 20.0, 40.0)
CONFORMAL_LEVEL = 0.90

SAMPLING_HOURS = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
ud.set_context('notebook')

# %% [markdown]
# ## Step 1 · Read the unified dataset

# %%
frames = {s: ud.load_unified(DATA_DIR, s) for s in STATIONS}
df = frames[STATION]

print(f'{STATION}: {len(df)} hourly slots, '
      f'{df.index.min()} to {df.index.max()}')
print(f'columns: {list(df.columns)}')
for col in (TARGET_PER_ERA, TARGET):
    print(f'  {col:18s} observed {df[col].notna().sum():6d} '
          f'({100 * df[col].notna().mean():.1f} %)')

# %% [markdown]
# ## Step 2 · What is actually there
#
# ### Per station

# %%
classification = ud.classify_stations(frames, freq=ud.ANALYSIS_FREQ)
display(classification[['eras', 'first_reading', 'last_reading', 'span_days',
                        'observed_days', 'coverage_of_span_%',
                        'coverage_of_archive_%']])
classification.to_csv(f'{OUTPUT_DIR}/U_04_classification.csv')

# %% [markdown]
# **The network did not survive to the changeover.** Two of the three legacy
# stations stopped recording long before the February 2025 replacement, so the
# archive narrows from three instrumented locations to one in stages rather
# than at a single moment.

# %%
ud.plot_station_completeness(
    frames, column='inc',
    title='Daily availability of the inclinometer, by station',
    save_path=OUTPUT_DIR, filename=f'U_F03_station_completeness')
plt.show()

# %% [markdown]
# ### Per era, for the target station

# %%
era_table = ud.era_profile(df, freq=ud.ANALYSIS_FREQ)
display(era_table)
era_table.to_csv(f'{OUTPUT_DIR}/U_05_era_profile_{STATION}.csv')

# %% [markdown]
# ### Per channel, along the record

# %%
availability = ud.channel_availability(df)
ud.plot_channel_availability(
    availability,
    title=f'{STATION}: daily availability of every channel',
    save_path=OUTPUT_DIR, filename=f'U_F04_channel_availability')
plt.show()

monthly = ud.monthly_completeness(frames, column='inc')
monthly.to_csv(f'{OUTPUT_DIR}/U_06_monthly_completeness.csv')

# %% [markdown]
# ### The calendar view
#
# Completeness as a year-by-day-of-year map, which is where a seasonal pattern
# in the outages would show itself.

# %%
calendar = ud.completeness_calendar(df[TARGET_PER_ERA])
ud.plot_completeness_calendar(
    calendar,
    title=f'{STATION}: fraction of each day observed',
    save_path=OUTPUT_DIR, filename=f'U_F05_calendar')
plt.show()
calendar.round(3).to_csv(f'{OUTPUT_DIR}/U_07_completeness_calendar.csv')

# %% [markdown]
# ## Step 3 · What is missing, and why
#
# ### Every gap, individually

# %%
gaps = ud.gap_table(df[TARGET_PER_ERA], dt_hours=SAMPLING_HOURS)
bands = ud.gap_bands(gaps)
display(bands)
gaps.to_csv(f'{OUTPUT_DIR}/U_08_gap_table.csv', index=False)
bands.to_csv(f'{OUTPUT_DIR}/U_09_gap_bands.csv')

print(f'{len(gaps)} gaps in all; the longest is '
      f'{gaps["hours"].max():.0f} h ({gaps["hours"].max() / 24:.0f} days)')

# %% [markdown]
# ### The gaps that dominate

# %%
big = gaps[gaps['days'] > 3].copy()
display(big[['start', 'end', 'days', 'season', 'context_before_h',
             'context_after_h']].reset_index(drop=True))
big.to_csv(f'{OUTPUT_DIR}/U_10_major_gaps.csv', index=False)

print(f'{len(big)} gaps longer than three days hold '
      f'{100 * big["hours"].sum() / gaps["hours"].sum():.1f} % of all missing '
      f'hours.')

# %%
ud.plot_gap_anatomy(
    gaps, bands=bands,
    title=f'{STATION}: where the missing hours are',
    save_path=OUTPUT_DIR, filename=f'U_F06_gap_anatomy')
plt.show()

# %% [markdown]
# ### Why they are missing
#
# The taxonomy is a design input, not a label. A gap missing at random can be
# filled from its neighbours; a gap whose absence tracks an observed covariate
# can be filled from that covariate; a gap that is an outage takes the
# covariates with it and can be filled from neither.

# %%
mechanism = ud.missingness_mechanism(df, target='inc')
display(mechanism)
mechanism.to_csv(f'{OUTPUT_DIR}/U_11_missingness_mechanism.csv')

print(f'Missing: {mechanism.attrs["missing_hours"]:.0f} h '
      f'({mechanism.attrs["missing_%"]:.1f} % of the archive span)')

# %% [markdown]
# The current era is tested separately, because solar radiation and wall
# temperature do not exist before 2025-02-21 and would otherwise be counted as
# missing for seven years.

# %%
current = df.loc[df['era'] == 'current']
mechanism_current = ud.missingness_mechanism(current, target='inc')
display(mechanism_current)
mechanism_current.to_csv(f'{OUTPUT_DIR}/U_12_missingness_current.csv')

# %% [markdown]
# ## Step 4 · What can be filled, and how well
#
# Six fillers of increasing ambition, measured against gaps of known length
# injected where the record is complete. No filler sees the values it is asked
# to reproduce.
#
# The range matters because no single method is right at every length:
# interpolation is unbeatable across an hour and meaningless across a month,
# while a seasonal model knows nothing useful about an hour and is most of what
# is left after a week.

# %%
bench = ud.benchmark_fillers(
    df, TARGET_PER_ERA, gap_lengths=GAP_LENGTHS, methods=FAST_METHODS,
    drivers=DRIVERS, n_trials=N_TRIALS, seed=0,
    min_context_hours=48, dt_hours=SAMPLING_HOURS)

display(bench.pivot(index='gap_hours', columns='method', values='MAE_mdeg'))
bench.to_csv(f'{OUTPUT_DIR}/U_13_filler_benchmark.csv', index=False)

signal_sd = float(df[TARGET_PER_ERA].std())
print(f'For scale, the signal standard deviation is {signal_sd:.1f} mdeg.')

# %% [markdown]
# ### The uncertainty that goes with a filled value
#
# A mean absolute error says how wrong a filler is on average. It does not say
# how wrong a particular filled value might be, and that is what a downstream
# consumer needs in order to weight or exclude it. The per-trial errors are
# therefore treated as a calibration sample and converted directly into an
# interval, with no distributional assumption — these errors are neither
# Gaussian nor independent, and assuming either would understate them.

# %%
conformal = ud.conformal_bands(bench, level=CONFORMAL_LEVEL)
display(conformal.pivot(index='gap_hours', columns='method',
                        values='half_width_mdeg'))
conformal.to_csv(f'{OUTPUT_DIR}/U_14_conformal_bands.csv', index=False)

# %%
ud.plot_filler_benchmark(
    bench, conformal=conformal,
    title='Filler error against gap length, and the calibrated interval',
    save_path=OUTPUT_DIR, filename=f'U_F07_filler_benchmark')
plt.show()

# %% [markdown]
# ### The driver filler, and why its score is not to be believed
#
# The benchmark injects gaps where the record is complete, so the drivers are
# present throughout every injected gap. Step 3 established that in a *real*
# gap they are not: every channel fails together. The driver filler is
# therefore being scored under conditions that never occur in production.
#
# This is the exchangeability assumption behind any conformal guarantee, stated
# concretely and failing.

# %%
driver_rows = bench[bench['method'] == 'drivers']
best_other = (bench[bench['method'] != 'drivers']
              .groupby('gap_hours')['MAE_mdeg'].min())
comparison = pd.DataFrame({
    'drivers_MAE': driver_rows.set_index('gap_hours')['MAE_mdeg'],
    'best_other_MAE': best_other,
})
comparison['drivers_advantage_%'] = (
    100 * (comparison['best_other_MAE'] - comparison['drivers_MAE'])
    / comparison['best_other_MAE']).round(1)
comparison['available_in_a_real_gap'] = False
display(comparison)
comparison.to_csv(f'{OUTPUT_DIR}/U_15_driver_illusion.csv')

# %% [markdown]
# ## Step 5 · The policies
#
# A policy assigns one filler to each gap-length band, or refuses the band. A
# refusal is the useful output, not a failure: it names the holes that have to
# stay holes.

# %% [markdown]
# **The driver filler is measured and then refused.** It cannot run on a real
# gap, so allowing a policy to select it would credit the fill with an accuracy
# the fill does not have — in practice it would degrade silently to
# interpolation while still being labelled `drivers`. The policy therefore
# chooses among the methods that can actually run, and reports what excluding
# the best benchmark score costs at each band.

# %%
policies = ud.band_policies(bench, TOLERANCES)
policy_frames = []
for label, policy in policies.items():
    entry = policy[['max_gap_h', 'method', 'MAE_mdeg', 'benchmark_best',
                    'benchmark_best_MAE', 'cost_of_availability_mdeg']].copy()
    entry['policy'] = label
    policy_frames.append(entry)
    print(f'\n=== {label} ===')
    display(policy[['max_gap_h', 'method', 'MAE_mdeg', 'benchmark_best',
                    'cost_of_availability_mdeg', 'decision']])

pd.concat(policy_frames).to_csv(f'{OUTPUT_DIR}/U_16_policies.csv')

# %% [markdown]
# ### What the availability constraint costs

# %%
headline = policies[f'tolerance {max(TOLERANCES):g} mdeg']
cost = headline[['benchmark_best', 'benchmark_best_MAE', 'best_usable',
                 'best_usable_MAE', 'cost_of_availability_mdeg']]
display(cost)
cost.to_csv(f'{OUTPUT_DIR}/U_15b_availability_cost.csv')
print(f'Excluded from selection: {headline.attrs["excluded"]}')

# %% [markdown]
# ## Step 6 · How long a continuous stretch that buys
#
# The question the study exists to answer, posed as a trade. Every extra hour of
# continuous record costs invented data, and the frontier is the exchange rate.

# %%
frontier, filled = ud.stretch_frontier(
    df, TARGET, policies, drivers=None, dt_hours=SAMPLING_HOURS,
    conformal=conformal)

display(frontier[['stretch_days', 'annual_cycles', 'stretch_from',
                  'stretch_to', 'filled_in_stretch_h', 'fabricated_%',
                  'mean_half_width', 'max_half_width']])
frontier.to_csv(f'{OUTPUT_DIR}/U_17_stretch_frontier.csv')

print(f'\nWithout filling anything, the longest strictly continuous stretch is '
      f'{frontier.attrs["observed_stretch_days"]:.1f} days '
      f'({frontier.attrs["observed_stretch_from"]} to '
      f'{frontier.attrs["observed_stretch_to"]}).')

# %%
ud.plot_stretch_frontier(
    frontier,
    title='Continuous record gained against data invented',
    save_path=OUTPUT_DIR, filename=f'U_F08_stretch_frontier')
plt.show()

# %% [markdown]
# ### The same question, refusing the era join
#
# The frontier above lets a stretch cross the instrument changeover, which is
# legitimate only because `inc_comp_joined` has had the estimated offset
# applied. Refusing that estimate — because the two instruments differ in gain
# as well as in zero — confines every stretch to one era.
#
# Note that this is *not* the same as running the frontier on the per-era
# column. A run of present values spans the changeover either way; what changes
# is whether the values across it mean one thing. The restriction has to be
# imposed on the stretch, not inferred from the column.

# %%
frontier_era, filled_era = ud.stretch_frontier(
    df, TARGET, policies, drivers=None, dt_hours=SAMPLING_HOURS,
    conformal=conformal, boundary=ud.CURRENT_START)
display(frontier_era[['stretch_days', 'annual_cycles', 'stretch_from',
                      'stretch_to', 'fabricated_%']])
frontier_era.to_csv(f'{OUTPUT_DIR}/U_18_stretch_frontier_within_era.csv')

best_within = frontier_era['stretch_days'].max()
best_across = frontier['stretch_days'].max()
print(f'\nLongest stretch within one era : {best_within:.1f} days '
      f'({best_within / ud.YEAR_DAYS:.2f} annual cycles)')
print(f'Longest stretch across the join: {best_across:.1f} days '
      f'({best_across / ud.YEAR_DAYS:.2f} annual cycles)')
print(f'The join is worth {best_across - best_within:.1f} days.')

# %% [markdown]
# ## Step 7 · The recommended stretch
#
# The policy chosen is the one buying the most continuous record at a
# fabricated share the benchmark supports.

# %%
best_label = frontier['stretch_days'].idxmax()
best_row = frontier.loc[best_label]
best = filled[best_label]

print(f'Recommended policy : {best_label}')
print(f'Stretch            : {best_row["stretch_from"]} to '
      f'{best_row["stretch_to"]}')
print(f'Length             : {best_row["stretch_days"]:.1f} days '
      f'({best_row["annual_cycles"]:.2f} annual cycles)')
print(f'Fabricated within  : {best_row["fabricated_%"]:.2f} % '
      f'({best_row["filled_in_stretch_h"]:.0f} h)')
print(f'Interval half-width: mean {best_row["mean_half_width"]:.2f} mdeg, '
      f'max {best_row["max_half_width"]:.2f} mdeg '
      f'at {CONFORMAL_LEVEL:.0%} nominal')

stretch = best.loc[best_row['stretch_from']:best_row['stretch_to']]
display(stretch['method'].value_counts().to_frame('hours'))
stretch.to_csv(f'{OUTPUT_DIR}/U_19_recommended_stretch.csv')

# %% [markdown]
# ### The stretch, drawn with its provenance and its uncertainty

# %%
fig, axes = plt.subplots(2, 1, figsize=ud.figsize(11, 5.2), sharex=True,
                         height_ratios=[3, 1])
observed = stretch['source'] == 'observed'
axes[0].plot(stretch.index, stretch['value'], lw=0.5, color='0.55',
             label='value')
axes[0].fill_between(
    stretch.index, stretch['value'] - stretch['half_width'].fillna(0),
    stretch['value'] + stretch['half_width'].fillna(0),
    color='#A5202B', alpha=0.35, lw=0,
    label=f'{CONFORMAL_LEVEL:.0%} interval on filled values')
axes[0].scatter(stretch.index[~observed], stretch['value'][~observed], s=2,
                color='#A5202B', label='imputed', zorder=3)
axes[0].set_ylabel('mdeg')
axes[0].legend(fontsize='small', ncol=3)

axes[1].fill_between(stretch.index, 0, (~observed).astype(int), step='mid',
                     color='#A5202B', lw=0)
axes[1].set_ylim(0, 1)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['obs', 'imp'])
axes[1].set_xlabel('')

fig.suptitle(f'The recommended stretch: {best_row["stretch_days"]:.0f} days, '
             f'{best_row["fabricated_%"]:.1f} % imputed')
fig.savefig(f'{OUTPUT_DIR}/U_F09_recommended_stretch.png', dpi=150,
            bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 8 · Verdicts

# %%
verdict_rows = [
    {'question': 'how many stations, and for how long?',
     'answer': '; '.join(
         f'{s}: {classification.loc[s, "observed_days"]:.0f} observed days, '
         f'last reading {str(classification.loc[s, "last_reading"])[:10]}'
         for s in STATIONS)},
    {'question': 'how complete is the target station?',
     'answer': f'{classification.loc[STATION, "observed_days"]:.0f} of '
               f'{classification.loc[STATION, "span_days"]:.0f} days '
               f'({classification.loc[STATION, "coverage_of_archive_%"]:.1f} %)'},
    {'question': 'where are the missing hours?',
     'answer': f'{len(gaps)} gaps; '
               f'{bands.loc["> 30 d", "share_of_missing_%"]:.1f} % of all '
               f'missing hours sit in the '
               f'{int(bands.loc["> 30 d", "n_gaps"])} gaps over 30 days'},
    {'question': 'what is the missingness mechanism?',
     'answer': 'outage-driven (MNAR): every covariate is missing whenever the '
               'target is, so no covariate-conditioned method can fill a real '
               'gap'},
    {'question': 'best filler, short gaps',
     'answer': '; '.join(
         f'{int(h)} h: '
         f'{bench[bench["gap_hours"] == h].set_index("method")["MAE_mdeg"].idxmin()} '
         f'({bench[bench["gap_hours"] == h]["MAE_mdeg"].min():.2f} mdeg)'
         for h in (1, 6, 24) if (bench['gap_hours'] == h).any())},
    {'question': 'best filler, long gaps',
     'answer': '; '.join(
         f'{int(h)} h: '
         f'{bench[bench["gap_hours"] == h].set_index("method")["MAE_mdeg"].idxmin()} '
         f'({bench[bench["gap_hours"] == h]["MAE_mdeg"].min():.2f} mdeg)'
         for h in (168, 720, 1080) if (bench['gap_hours'] == h).any())},
    {'question': 'the driver filler',
     'answer': 'scores well on injected gaps and is unusable on real ones, '
               'because the drivers fail with the target — the exchangeability '
               'assumption behind the benchmark, failing'},
    {'question': 'longest stretch, nothing filled',
     'answer': f'{frontier.attrs["observed_stretch_days"]:.1f} days '
               f'({frontier.attrs["observed_stretch_from"][:10]} to '
               f'{frontier.attrs["observed_stretch_to"][:10]})'},
    {'question': 'longest stretch, after imputation',
     'answer': f'{best_row["stretch_days"]:.1f} days '
               f'({best_row["annual_cycles"]:.2f} cycles) under {best_label}, '
               f'{best_row["fabricated_%"]:.2f} % of it invented'},
    {'question': 'what still cannot be filled',
     'answer': f'{int((big["hours"] > max(TOLERANCES) * 0 + 1080).sum())} gaps '
               f'exceed the longest benchmarked length; the '
               f'{big["days"].max():.0f}-day outage is unbridgeable at any '
               f'tolerance'},
]
verdicts = pd.DataFrame(verdict_rows).set_index('question')
display(verdicts)
verdicts.to_csv(f'{OUTPUT_DIR}/U_20_verdicts.csv')
