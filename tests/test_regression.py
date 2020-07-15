"""
Test regressions with fixed seed
expected files are in the "expected" folder
"""

import unittest
import os
import shutil
import tempfile
import sciris as sc
import synthpops as sp

class TestRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.resultdir = tempfile.TemporaryDirectory().name

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.resultdir, ignore_errors=True)

    def test_regression_make_population(self):
        filename = os.path.join(self.resultdir,'pop.json')
        seed = 1001
        pop = sp.make_population(n=1001, rand_seed=seed, generate=True)
        # if default sort order is not concerned:
        # pop = dict(sorted(pop.items(), key=lambda x: x[0]))
        sc.savejson(filename, pop, indent=2)
        self.check_result(filename)

    def check_result(self, actual_file, expected_file = None):
        if not os.path.isfile(actual_file):
            raise FileNotFoundError(actual_file)
        if expected_file is None:
            expected_file = os.path.join(os.path.join(os.path.dirname(__file__), 'expected'), os.path.basename(actual_file))
        if not os.path.isfile(expected_file):
            raise FileNotFoundError(expected_file)
        expected = sc.loadjson(expected_file)
        actual = sc.loadjson(actual_file)
        self.assertEqual(actual, expected)

