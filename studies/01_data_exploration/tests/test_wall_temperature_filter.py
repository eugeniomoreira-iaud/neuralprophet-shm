"""Tests for auditable wall-temperature impulse filtering."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

import de_lib as de  # noqa: E402


class WallTemperatureFilterTests(unittest.TestCase):
    def test_changes_only_flagged_samples_after_start(self):
        """A spike inside the segment is interpolated; earlier data is untouched."""
        index = pd.date_range("2026-06-18 00:00", periods=7, freq="20min")
        series = pd.Series([100.0, 6.0, 10.0, 11.0, 50.0, 13.0, 14.0], index=index)
        original = series.copy()

        filtered, flags = de.filter_impulsive_segment(
            series,
            start=index[2],
            window_size=3,
            n_sigmas=3.0,
            floor=10.0,
        )

        self.assertEqual(filtered.loc[index[0]], 100.0)
        self.assertEqual(filtered.loc[index[4]], 12.0)
        self.assertEqual(flags[flags].index.tolist(), [index[4]])
        self.assertEqual(flags.dtype, bool)
        pd.testing.assert_series_equal(series, original)

    def test_preserves_original_gaps(self):
        """Filtering an observed spike must not fill a pre-existing missing slot."""
        index = pd.date_range("2026-06-18 00:00", periods=7, freq="20min")
        series = pd.Series(
            [10.0, np.nan, 12.0, 13.0, 50.0, 15.0, 16.0],
            index=index,
        )

        filtered, flags = de.filter_impulsive_segment(
            series,
            start=index[0],
            window_size=3,
            n_sigmas=3.0,
            floor=10.0,
        )

        self.assertTrue(pd.isna(filtered.loc[index[1]]))
        self.assertEqual(filtered.loc[index[4]], 14.0)
        self.assertTrue(bool(flags.loc[index[4]]))

    def test_does_not_bridge_a_gap_to_classify_or_replace_a_boundary_value(self):
        """A run boundary cannot borrow context from the opposite side of a gap."""
        index = pd.date_range("2026-06-18 00:00", periods=7, freq="20min")
        series = pd.Series(
            [10.0, 11.0, np.nan, np.nan, 80.0, 12.0, 13.0],
            index=index,
        )

        filtered, flags = de.filter_impulsive_segment(
            series,
            start=index[0],
            window_size=3,
            n_sigmas=3.0,
            floor=10.0,
        )

        self.assertEqual(filtered.loc[index[4]], 80.0)
        self.assertFalse(bool(flags.loc[index[4]]))
        self.assertTrue(filtered.loc[index[2:3]].isna().all())

    def test_export_carries_filter_provenance(self):
        """Archive exports must include filtered values and their provenance flag."""
        wide = pd.DataFrame(
            {
                "era": ["current"],
                "n_twall_filtered": [21.5],
                "twall_spike": [False],
            },
            index=pd.to_datetime(["2026-06-18"]),
        )

        self.assertEqual(
            de.export_columns(wide),
            ["era", "n_twall_filtered", "twall_spike"],
        )

    def test_target_view_exposes_filtered_wall_temperature_separately(self):
        """The analysis view must preserve corrected and filtered wall temperatures."""
        index = pd.to_datetime(["2025-02-20 20:00", "2026-06-18 10:00"])
        wide = pd.DataFrame(index=index)
        wide["era"] = ["legacy", "current"]
        for channel in ("batt", "tair", "rh"):
            wide[f"st02_{channel}_ok"] = [1.0, np.nan]
            wide[f"n_{channel}_ok"] = [np.nan, 2.0]
        wide["n_sr_ok"] = [np.nan, 500.0]
        wide["n_twall_ok"] = [np.nan, 85.0]
        wide["n_twall_filtered"] = [np.nan, 39.0]
        wide["twall_spike"] = [False, True]
        wide["inc"] = [10.0, 11.0]

        view = de.target_view(wide)

        self.assertEqual(view.loc[index[1], "twall"], 85.0)
        self.assertEqual(view.loc[index[1], "twall_filtered"], 39.0)
        self.assertTrue(bool(view.loc[index[1], "twall_spike"]))


if __name__ == "__main__":
    unittest.main()
