"""
Tests for the tier and block machinery added to pc_lib.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_tiers.py

The frames here are synthetic. A deliberate multi-day hole splits the record so
that block-boundary behaviour is exercised rather than assumed.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def build_frame():
    """
    A synthetic frame with one long hole and one short one.

    The record runs from 1 January 2025 to 1 January 2026 on the hourly grid.
    A 10-day hole in June splits the current era into two blocks, both of them
    comfortably longer than the 20-day reporting floor. A 3-hour hole in May is
    short enough to be absorbed, because max_gap_hours is 6.
    """
    index = pd.date_range('2025-01-01', '2026-01-01', freq='h')
    df = pd.DataFrame({
        'inc_comp': np.linspace(0.0, 100.0, len(index)),
        'tair_gs': np.linspace(0.0, 50.0, len(index)),
        'sr_gs': np.linspace(0.0, 900.0, len(index)),
        'tair_str': np.linspace(0.0, 50.0, len(index)),
        'sr_str': np.linspace(0.0, 900.0, len(index)),
    }, index=index)
    df.loc['2025-06-01':'2025-06-11', :] = np.nan
    df.loc['2025-05-01 00:00':'2025-05-01 02:00', :] = np.nan
    # The on-structure radiation channel exists only in the current era.
    df.loc[:'2025-02-20 23:00', 'sr_str'] = np.nan
    return df


def test_short_hole_absorbed_long_hole_splits():
    """A 3-hour interruption stays inside a block; a 10-day one ends it."""
    df = build_frame()
    inventory = pc.tier_blocks(
        df, {'T1': ['inc_comp', 'tair_gs', 'sr_gs']},
        era_boundary=pd.Timestamp('2025-02-21'))
    current = inventory[inventory['era'] == 'current']
    assert len(current) == 2, f'expected 2 blocks, got {len(current)}'
    assert current['days'].max() > 150, 'the post-June block should be long'
    assert current['days'].min() > 20, 'both blocks should clear min_days'


def test_no_block_crosses_the_era_boundary():
    """Every block lies wholly inside one era."""
    df = build_frame()
    boundary = pd.Timestamp('2025-02-21')
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs', 'sr_gs']},
                               era_boundary=boundary)
    for _, row in inventory.iterrows():
        if row['era'] == 'legacy':
            assert row['end'] < boundary
        else:
            assert row['start'] >= boundary


def test_tier_requiring_sr_str_is_current_era_only():
    """A tier requiring the short channel cannot reach the legacy era."""
    df = build_frame()
    inventory = pc.tier_blocks(
        df, {'P0': ['inc_comp', 'sr_str'], 'P1': ['inc_comp', 'sr_gs']},
        era_boundary=pd.Timestamp('2025-02-21'))
    p0 = inventory[inventory['tier'] == 'P0']
    assert (p0['era'] == 'current').all(), 'P0 leaked into the legacy era'
    assert 'legacy' in set(inventory[inventory['tier'] == 'P1']['era'])


def test_tier_index_is_the_union_of_its_blocks():
    """The union index covers every block and nothing outside them."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    index = pc.tier_index(inventory, 'T1')
    assert index.is_monotonic_increasing
    assert index.is_unique
    # No timestamp from the long June hole survives.
    assert not index.isin(pd.date_range('2025-06-03', '2025-06-09',
                                        freq='h')).any()


def test_differences_are_not_taken_across_a_block_boundary():
    """
    The step manufactured by differencing across the June hole must not appear.

    The synthetic series rises by a constant amount per hour, so every honest
    first difference is that same constant. A difference taken across the
    10-day hole would be far larger, and this test fails if one survives.
    """
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    diffs = pc.tier_differences(df, inventory, 'T1', cols=['inc_comp'])
    step = diffs['inc_comp'].dropna()
    assert len(step) > 1000, 'too few differences survived'
    assert step.max() < step.median() * 2.0, (
        f'a boundary step survived: max {step.max():.4f} '
        f'against median {step.median():.4f}')


def test_default_differencing_skips_non_numeric_columns():
    """
    The harmonised frame carries non-numeric channels beside the measurements.

    Differencing one raises TypeError rather than returning anything useful, so
    the cols=None default must exclude them instead of failing.
    """
    df = build_frame()
    df['note'] = 'text'
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    diffs = pc.tier_differences(df, inventory, 'T1')
    assert 'note' not in diffs.columns
    assert 'inc_comp' in diffs.columns


def test_longest_block_is_the_longest():
    """The scan window selector returns the longest block, not the first."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    block = pc.longest_block(inventory, 'T1')
    tier_rows = inventory[inventory['tier'] == 'T1']
    assert block['days'] == tier_rows['days'].max()


def test_missing_tier_returns_empty_rather_than_raising():
    """A tier whose channels never coexist is reported, not an exception."""
    df = build_frame()
    df['never'] = np.nan
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'never']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    assert len(inventory) == 0
    assert list(inventory.columns) == ['tier', 'era', 'block', 'start', 'end',
                                       'days', 'coverage_%']


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all tier tests passed')
