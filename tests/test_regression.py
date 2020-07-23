"""
Test regressions with fixed seed
expected files are in the "expected" folder
the filename has pattern pop_{n}_seed{seed}.json
"""

import unittest
import os
import shutil
import tempfile
import sciris as sc
import synthpops as sp
import numpy as np
from scipy.spatial import distance
from examples import plot_age_mixing_matrices

class TestRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.resultdir = tempfile.TemporaryDirectory().name

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.resultdir, ignore_errors=True)

    def test_regression_make_population(self):
        n = 10001
        seed = 1001
        filename = os.path.join(self.resultdir,f'pop_{n}_seed{seed}.json')
        #test parameter should be specific
        pop = sp.make_population(n=n,
                                 max_contacts=None,
                                 with_industry_code= False,
                                 with_facilities= False,
                                 use_two_group_reduction= False,
                                 average_LTCF_degree= 20,
                                 rand_seed=seed,
                                 generate=True)
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
        expected = self.cast_uid_toINT(sc.loadjson(expected_file))
        actual = self.cast_uid_toINT(sc.loadjson(actual_file))
        self.check_similarity(actual, expected)

        # if test failed, look at the figures
        #fig = plot_age_mixing_matrices.test_plot_generated_contact_matrix(population=expected, title_prefix="expected_")
        #fig.show()
        #fig = plot_age_mixing_matrices.test_plot_generated_contact_matrix(population=actual, title_prefix="actual_")
        #fig.show()

    def check_similarity(self, actual, expected):
        """
        Compare two population dictionaries using contact matrix
        Assuming the canberra distance should be close to zero
        """
        for code in ['H', 'W', 'S']:
            for option in ['density', 'frequency']:
                print(f"\ncheck:{code} with {option}")
                actual_matrix = sp.calculate_contact_matrix(actual, density_or_frequency=option, setting_code=code)
                expected_matrix = sp.calculate_contact_matrix(expected, density_or_frequency=option, setting_code=code)
                # calculate Canberra distance
                # assuming they should round to 0
                d = distance.canberra(actual_matrix.flatten(), expected_matrix.flatten())
                self.assertEqual(round(d), 0, f"actual distance for {code}/{option} is {str(round(d))}, "
                                              f"you need to uncommented line 55 to plot the density matrix and investigate!")

    def cast_uid_toINT(self, dict):
        return {int(key): val for key, val in dict.items()}