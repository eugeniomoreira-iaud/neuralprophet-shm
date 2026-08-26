"""
Guards that the study folder never again claims a result it does not hold.

These are repository hygiene tests rather than numerical ones: they assert that
no code lives outside shmlib, and that no artefact is named in prose without a
file behind it.
"""
import re
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPORT = STUDY / 'report' / 'neuralprophet_inclination_prediction_report.tex'
README = STUDY / 'README.md'


class TestNoCodeOutsideShmlib(unittest.TestCase):

    def test_only_the_notebook_source_is_a_python_file_in_the_study_root(self):
        found = sorted(p.name for p in STUDY.glob('*.py'))
        self.assertEqual(
            found, ['neuralprophet_inclination_prediction_study.py'],
            'A study folder holds its notebook and nothing else executable; '
            'every function belongs in studies/shmlib/.')


class TestReportClaimsAreSupported(unittest.TestCase):

    def test_report_does_not_assert_unproduced_results(self):
        text = REPORT.read_text(encoding='utf-8')
        for phrase in ('definitive 48-hour limit', 'pinball quantiles',
                       'horizon-adaptive block-bootstrapping'):
            self.assertNotIn(
                phrase, text,
                f'The report asserts {phrase!r} with no artefact behind it.')

    def test_every_included_graphic_exists(self):
        text = REPORT.read_text(encoding='utf-8')
        outputs = STUDY / 'outputs'
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            stem = Path(name).stem
            self.assertTrue(
                list(outputs.glob(stem + '.*')),
                f'The report includes {name} but outputs/ holds no such file.')


class TestReadmeMatchesTheFolder(unittest.TestCase):

    def test_readme_names_only_artefacts_that_exist(self):
        text = README.read_text(encoding='utf-8')
        outputs = STUDY / 'outputs'
        for name in re.findall(r'`(NP_F?\d\d[A-Za-z0-9_]*)', text):
            self.assertTrue(
                list(outputs.glob(name + '*')),
                f'README names {name}, which outputs/ does not contain.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
