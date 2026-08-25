"""Regression tests for span highlights in the generated report figures."""

from pathlib import Path
import unittest


OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'
# SVG's default fill is black, so Matplotlib serializes a black 5%-opaque patch
# with only the non-default opacity declaration.
BLACK_FIVE_PERCENT = 'style="opacity: 0.05"'


class ReportSpanHighlightTests(unittest.TestCase):
    """The report uses one visual treatment for every highlighted span."""

    def test_every_report_span_uses_black_at_five_percent_opacity(self):
        """Every generated span is black at 5% opacity, with none omitted."""
        expected_spans = {
            'DE_F04_st02_inclination.svg': 1,
            'DE_F07_st02_inclination_cleaned.svg': 1,
            'DE_F14_st02_anomaly_channels.svg': 4,
            'DE_F16_st02_twall_raw.svg': 1,
            'DE_F17_st02_twall_filtered.svg': 1,
        }

        for filename, expected_count in expected_spans.items():
            with self.subTest(filename=filename):
                svg = (OUTPUT_DIR / filename).read_text(encoding='utf-8')
                self.assertEqual(svg.count(BLACK_FIVE_PERCENT), expected_count)


if __name__ == '__main__':
    unittest.main()
