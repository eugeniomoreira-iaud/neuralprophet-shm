"""
Tests for the solar geometry added to pc_lib.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_solar.py

These tests build their own timestamps and never read the archive or data/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def test_elevation_peaks_near_solar_noon():
    """The daily maximum elevation must fall within an hour of solar noon."""
    index = pd.date_range('2025-06-21', periods=24, freq='h')
    elevation = pc.solar_elevation(index)
    peak_hour = int(elevation.idxmax().hour)
    solar_noon = float(pc.solar_noon_utc(pd.DatetimeIndex(['2025-06-21'])).iloc[0])
    assert abs(peak_hour - solar_noon) <= 1.0, (
        f'peak at {peak_hour} h, solar noon at {solar_noon:.2f} h')


def test_summer_noon_higher_than_winter_noon():
    """At this latitude the June sun must stand well above the December sun."""
    june = pc.solar_elevation(pd.DatetimeIndex(['2025-06-21 12:00'])).iloc[0]
    december = pc.solar_elevation(pd.DatetimeIndex(['2025-12-21 12:00'])).iloc[0]
    assert june > december + 30.0, f'June {june:.1f}, December {december:.1f}'


def test_elevation_within_physical_bounds():
    """Elevation is an angle and cannot leave the range it is defined on."""
    index = pd.date_range('2025-01-01', periods=24 * 365, freq='h')
    elevation = pc.solar_elevation(index)
    assert elevation.between(-90.0, 90.0).all()
    assert elevation.notna().all()


def test_daylight_fraction_is_plausible():
    """Averaged over a year the sun is up for roughly half the hours."""
    index = pd.date_range('2025-01-01', periods=24 * 365, freq='h')
    fraction = float(pc.daylight_mask(index).mean())
    assert 0.45 < fraction < 0.55, f'daylight fraction {fraction:.3f}'


def test_daylight_longer_in_june_than_december():
    """Day length must vary with the season, which a fixed clock window cannot do."""
    june = float(pc.daylight_mask(
        pd.date_range('2025-06-21', periods=24, freq='h')).sum())
    december = float(pc.daylight_mask(
        pd.date_range('2025-12-21', periods=24, freq='h')).sum())
    assert june - december >= 5.0, f'June {june} h, December {december} h'


def test_mask_is_boolean_and_aligned():
    """The mask must be usable directly as a selector on the analysis frame."""
    index = pd.date_range('2025-03-01', periods=48, freq='h')
    mask = pc.daylight_mask(index)
    assert mask.dtype == bool
    assert mask.index.equals(index)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all solar tests passed')
