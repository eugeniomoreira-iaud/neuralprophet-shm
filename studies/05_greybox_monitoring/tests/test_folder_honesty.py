"""
Guards that the study folder never claims a result it does not hold.

Run from studies/:  python 05_greybox_monitoring/tests/test_folder_honesty.py

Repository hygiene rather than numerics: no code outside shmlib, every
graphic and table body the report includes exists on disk, and, once the
study is complete, no section is still marked pending.
"""
import re
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPORT = STUDY / 'report' / 'greybox_monitoring_report.tex'
README = STUDY / 'README.md'
OUTPUTS = STUDY / 'outputs'

# Flipped to True in the final task, when every artefact the README names
# must exist and no section may remain pending.
STUDY_COMPLETE = False


class TestNoCodeOutsideShmlib(unittest.TestCase):

    def test_only_the_notebook_source_is_a_python_file_in_the_study_root(self):
        found = sorted(p.name for p in STUDY.glob('*.py'))
        self.assertEqual(found, ['greybox_monitoring_study.py'])


class TestReportClaimsAreSupported(unittest.TestCase):

    def test_every_included_graphic_exists(self):
        text = REPORT.read_text(encoding='utf-8')
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            stem = Path(name).stem
            self.assertTrue(list(OUTPUTS.glob(stem + '.*')),
                            f'report includes {name}; outputs/ has no such file')

    def test_every_input_table_body_exists(self):
        text = REPORT.read_text(encoding='utf-8')
        for name in re.findall(r'\\input\{\.\./outputs/([^}]+)\}', text):
            self.assertTrue((OUTPUTS / name).exists(),
                            f'report inputs {name}; outputs/ has no such file')

    def test_no_section_is_pending_once_complete(self):
        if not STUDY_COMPLETE:
            self.skipTest('study in progress; pending sections allowed')
        text = REPORT.read_text(encoding='utf-8')
        self.assertNotIn(r'\pending{', text)


class TestReadmeMatchesTheFolder(unittest.TestCase):

    def test_readme_names_only_artefacts_that_exist(self):
        if not STUDY_COMPLETE:
            self.skipTest('study in progress; README lists planned artefacts')
        text = README.read_text(encoding='utf-8')
        for name in re.findall(r'`(GM_F?\d\d[A-Za-z0-9_]*)', text):
            self.assertTrue(list(OUTPUTS.glob(name + '*')),
                            f'README names {name}; outputs/ does not contain it')


if __name__ == '__main__':
    unittest.main(verbosity=2)
