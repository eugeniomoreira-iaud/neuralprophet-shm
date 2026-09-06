"""
Guards that the study folder never claims a result it does not hold.

Run from studies/:  python 05_greybox_monitoring/tests/test_folder_honesty.py

Repository hygiene rather than numerics: no code outside shmlib, every
graphic and table body the report includes exists on disk, and, once the
study is complete, no section is still marked pending. A further guard,
``TestEveryGraphicComesFromTheNotebook``, checks that every figure and table
body the report references is one the paired notebook's own code actually
writes, rather than a stale file left over from an earlier run or a
hand-typed reference nothing produces.
"""
import re
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPORT = STUDY / 'report' / 'greybox_monitoring_report.tex'
README = STUDY / 'README.md'
OUTPUTS = STUDY / 'outputs'
NOTEBOOK = STUDY / 'greybox_monitoring_study.py'

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


def _notebook_artefact_names():
    """
    The filenames and table bodies the notebook writes, read from its own
    source text.

    Two literal patterns cover every artefact the report might reference:
    every ``filename='...'`` argument passed to one of the notebook's
    ``figures.*`` plotting calls, and every ``'GM_NN[a-z]?_body.tex'``
    string literal passed to one of its ``tables.write_table`` calls.
    Reading these straight out of the notebook's own source, rather than
    trusting a list maintained separately, is what lets this guard catch a
    report reference to a figure or table body the notebook does not
    actually produce — the report and the notebook can drift independently,
    but the notebook's source is always the ground truth for what it
    writes.

    Returns
    -------
    tuple of set
        ``(filenames, bodies)``: the stems saved through ``figures.*``
        calls, and the body filenames written through ``tables.write_table``
        calls.
    """
    text = NOTEBOOK.read_text(encoding='utf-8')
    filenames = set(re.findall(r"filename='([^']+)'", text))
    bodies = set(re.findall(r"'(GM_\d\d[a-z]?_body\.tex)'", text))
    return filenames, bodies


class TestEveryGraphicComesFromTheNotebook(unittest.TestCase):
    """
    Implements the global constraint "Every image in the report comes from
    the paired notebook" (user rule, 2026-09-06): a figure or table body the
    report shows must be one the notebook's own code actually writes, so
    that a reader who checks the notebook can always find where a number or
    a picture in the report came from.
    """

    def test_every_included_graphic_is_a_notebook_filename(self):
        filenames, _ = _notebook_artefact_names()
        text = REPORT.read_text(encoding='utf-8')
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            stem = Path(name).stem
            self.assertIn(stem, filenames,
                          f'report includes {stem}; a figure is produced by '
                          'the notebook or not at all')

    def test_every_input_table_body_is_a_notebook_body(self):
        _, bodies = _notebook_artefact_names()
        text = REPORT.read_text(encoding='utf-8')
        for name in re.findall(r'\\input\{\.\./outputs/([^}]+)\}', text):
            self.assertIn(name, bodies,
                          f'report inputs {name}; a figure is produced by '
                          'the notebook or not at all')

    def test_no_orphan_image_in_outputs(self):
        filenames, _ = _notebook_artefact_names()
        for path in OUTPUTS.glob('GM_F*.png'):
            self.assertIn(path.stem, filenames,
                          f'{path.name} sits in outputs/; a figure is '
                          'produced by the notebook or not at all')


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
