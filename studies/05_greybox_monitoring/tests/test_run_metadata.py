"""
Tests for tables.run_metadata (Study 05 Task 7.1).

Run from studies/:  python 05_greybox_monitoring/tests/test_run_metadata.py
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from shmlib import tables  # noqa: E402


class TestRunMetadata(unittest.TestCase):

    def test_parameter_rows_precede_version_rows_and_a_tuple_renders_as_str(self):
        parameters = {'window_end': '2026-06-30', 'limits_L': (1.0, 2.0)}
        versions = {'neuralprophet_version': '0.8.0', 'pandas_version': '2.2.2',
                    'numpy_version': '1.26.4', 'scipy_version': '1.13.1'}
        out = tables.run_metadata(parameters, versions=versions)

        self.assertEqual(list(out.columns), ['parameter', 'value'])
        self.assertEqual(len(out), 6)
        self.assertEqual(list(out['parameter']), [
            'window_end', 'limits_L', 'neuralprophet_version', 'pandas_version',
            'numpy_version', 'scipy_version'])
        self.assertEqual(out.loc[out['parameter'] == 'window_end', 'value'].item(),
                          '2026-06-30')
        self.assertEqual(out.loc[out['parameter'] == 'limits_L', 'value'].item(),
                          str((1.0, 2.0)))
        self.assertEqual(out.loc[out['parameter'] == 'neuralprophet_version',
                                  'value'].item(), '0.8.0')
        self.assertEqual(out.loc[out['parameter'] == 'scipy_version',
                                  'value'].item(), '1.13.1')


if __name__ == '__main__':
    unittest.main(verbosity=2)
