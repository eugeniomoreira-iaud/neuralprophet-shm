import sys, os
import pandas as pd
sys.path.insert(0, os.path.abspath('..'))
from shmlib import proxies, site

ARCHIVE_CSV = '../../data/interim/archive/gubbio_archive_20min.csv'
RESPONSE_COLUMN = 'inc_comp_cleaned'
ANALYSIS_FREQ = site.ANALYSIS_FREQ
SITE_TZ = site.SITE_TZ

response, _ = proxies.load_response(
    ARCHIVE_CSV, column=RESPONSE_COLUMN, honour_spike=True,
    freq=ANALYSIS_FREQ, tz=SITE_TZ, min_count=site.MIN_SAMPLES_PER_HOUR)

window = response.loc[site.to_utc(pd.DatetimeIndex(['2023-06-21']))[0]:]

target_hours = 7 * 24
consecutive = 0
start_idx = None

for idx, val in window.items():
    if pd.isna(val):
        consecutive = 0
        start_idx = None
    else:
        if start_idx is None:
            start_idx = idx
        consecutive += 1
        if consecutive == target_hours:
            print(f"Found good 7-day window starting at {start_idx}")
            break
