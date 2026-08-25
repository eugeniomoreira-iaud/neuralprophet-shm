"""
Tests for the block-aware extension of pc.target_relationship.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_target_relationship.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def build_frame(seed=0):
    """
    Two clean blocks separated by a long hole, with a known linear relationship.

    Inside each block the target is -3.0 times the driver plus noise, so a
    correct fit must recover a slope near -3.0 in both domains.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range('2025-03-01', periods=24 * 400, freq='h')
    driver = pd.Series(
        10.0 + 8.0 * np.sin(2 * np.pi * np.arange(len(index)) / 24.0)
        + rng.normal(0.0, 0.5, len(index)), index=index)
    target = -3.0 * driver + rng.normal(0.0, 0.5, len(index))
    df = pd.DataFrame({'inc_comp': target, 'tair_gs': driver})
    df.loc['2025-07-01':'2025-08-15', :] = np.nan
    return df


def test_default_call_is_unchanged():
    """Omitting blocks must reproduce today's behaviour and columns exactly."""
    df = build_frame()
    out = pc.target_relationship(df, 'tair', sources=('gs',))
    expected = ['quantity', 'source', 'domain', 'n', 'overlap_days', 'r',
                'slope_mdeg_per_unit', 'slope_se', 't', 'r2']
    assert list(out.columns) == expected, list(out.columns)
    assert set(out['domain']) == {'levels', 'differences'}
    assert 'tier' not in out.columns


def test_block_call_recovers_the_known_slope():
    """On clean blocks the fit must return the slope that built the data."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    out = pc.target_relationship(df, 'tair', sources=('gs',),
                                 blocks=inventory, tier='T1')
    levels = out[out['domain'] == 'levels'].iloc[0]
    assert abs(levels['slope_mdeg_per_unit'] + 3.0) < 0.1, levels
    assert levels['tier'] == 'T1'


def test_block_call_reports_fewer_rows_than_the_pooled_call():
    """Restricting to blocks must drop the hours inside the hole."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    pooled = pc.target_relationship(df, 'tair', sources=('gs',))
    tiered = pc.target_relationship(df, 'tair', sources=('gs',),
                                    blocks=inventory, tier='T1')
    pooled_n = int(pooled[pooled['domain'] == 'levels']['n'].iloc[0])
    tiered_n = int(tiered[tiered['domain'] == 'levels']['n'].iloc[0])
    assert tiered_n <= pooled_n


def test_differences_do_not_cross_the_hole():
    """
    A difference across the 45-day hole would wreck the differenced fit.

    With per-block differencing the differenced slope must still be near -3.0.
    """
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    out = pc.target_relationship(df, 'tair', sources=('gs',),
                                 blocks=inventory, tier='T1')
    diff = out[out['domain'] == 'differences'].iloc[0]
    assert abs(diff['slope_mdeg_per_unit'] + 3.0) < 0.2, diff


def test_empty_tier_returns_empty_frame():
    """A tier with no block yields no rows rather than an exception."""
    df = build_frame()
    empty = pd.DataFrame(columns=pc.BLOCK_COLUMNS)
    out = pc.target_relationship(df, 'tair', sources=('gs',),
                                 blocks=empty, tier='T1')
    assert len(out) == 0


def test_blocks_without_tier_is_rejected():
    """Passing blocks without naming a tier is a programming error, not a default."""
    df = build_frame()
    inventory = pc.tier_blocks(df, {'T1': ['inc_comp', 'tair_gs']},
                               era_boundary=pd.Timestamp('2025-02-21'))
    try:
        pc.target_relationship(df, 'tair', sources=('gs',), blocks=inventory)
    except ValueError:
        return
    raise AssertionError('expected ValueError when tier is omitted')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all target-relationship tests passed')
