"""
Tests for the acquisition-fault register added to pc_lib.

Run with:
    conda activate neuralprophet_env
    cd studies/proxy_comparison
    python tests/test_faults.py

The frames are synthetic and carry faults of known shape, so the register is
tested against events whose cause is known rather than against the archive.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pc_lib as pc


def build_frame(n=24 * 120):
    """A smooth diurnal record with no faults in it."""
    index = pd.date_range('2025-03-01', periods=n, freq='h')
    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / 24.0)
    return pd.DataFrame({
        'inc_comp': -30.0 + 5.0 * daily,
        'inc': -310.0 + 5.0 * daily,
        'tair': 18.0 + 4.0 * daily,
        'batt': 4.41 + 0.01 * daily,
    }, index=index)


def test_clean_record_flags_nothing():
    """A smooth record must produce an empty register, not false positives."""
    faults = pc.acquisition_faults(build_frame())
    assert len(faults) == 0, faults.head().to_string()


def test_single_hour_multichannel_spike_is_flagged_and_corroborated():
    """
    The signature of the 2025-05-10 event: three channels fail for one hour.

    The register must flag the hour, record that other channels moved with it,
    and record that the excursion reversed on the following hour.
    """
    df = build_frame()
    when = df.index[500]
    df.loc[when, 'inc_comp'] -= 250.0
    df.loc[when, 'inc'] -= 220.0
    df.loc[when, 'tair'] += 6.0
    df.loc[when, 'batt'] -= 0.08

    faults = pc.acquisition_faults(df)
    assert when in faults.index, faults.to_string()
    assert faults.loc[when, 'n_corroborating'] >= 2, faults.loc[when]
    assert bool(faults.loc[when, 'reverses_next_hour'])


def test_sustained_step_is_not_flagged_as_a_fault():
    """
    A genuine level shift must survive.

    A step changes one first difference and then stays, which is what a real
    structural or instrumental re-levelling looks like. Treating it as a
    single-hour fault would delete a real event, so the register must not.
    """
    df = build_frame()
    df.loc[df.index[600]:, 'inc_comp'] -= 40.0
    faults = pc.acquisition_faults(df)
    stepped = df.index[600]
    assert stepped not in faults.index, faults.to_string()


def test_mask_removes_only_the_flagged_hours():
    """Masking must blank the flagged hours and leave every other value alone."""
    df = build_frame()
    when = df.index[500]
    df.loc[when, ['inc_comp', 'inc', 'tair']] += [-250.0, -220.0, 6.0]

    faults = pc.acquisition_faults(df)
    masked = pc.mask_faults(df, faults, cols=['inc_comp'])
    assert masked.loc[when, 'inc_comp'] != masked.loc[when, 'inc_comp']  # NaN
    kept = df.index.difference(faults.index)
    assert masked.loc[kept, 'inc_comp'].equals(df.loc[kept, 'inc_comp'])
    # Other columns are untouched unless named.
    assert masked['tair'].equals(df['tair'])


def test_register_carries_the_evidence_columns():
    """The register is a justification, so it must carry the evidence."""
    df = build_frame()
    when = df.index[500]
    df.loc[when, ['inc_comp', 'inc', 'tair']] += [-250.0, -220.0, 6.0]
    faults = pc.acquisition_faults(df)
    for col in ('z_inc_comp', 'delta_mdeg', 'n_corroborating',
                'reverses_next_hour'):
        assert col in faults.columns, list(faults.columns)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('all fault-register tests passed')
