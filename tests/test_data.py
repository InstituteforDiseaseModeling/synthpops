print('Note -- these tests currently deprecated')

import numpy as np
import pandas as pd
import sciris as sc
import synthpops as sp
import os
import unittest
import pathlib

datadir = pathlib.Path(__file__).parent.parent.joinpath("synthpops/data").absolute()


class TestLocation(unittest.TestCase):
    """
    These tests need to be run from the root synthpops/tests folder, because some of the tests involve relative
    filepath assumptions based on that.
    """

    @classmethod
    def setUpClass(cls) -> None:
        sp.settings.datadir = sc.thisdir(__file__, 'testdata')

    @classmethod
    def tearDownClass(cls) -> None:
        sp.settings.datadir = datadir

    def minimal_test_str(self):
        test_str = """{
          "location_name": "test_location",
          "data_provenance_notices": ["notice1","notice2"],
          "reference_links": ["reference1","reference2"],
          "citations": ["citation1","citation2"],
          "parent": {},
          "population_age_distribution_16": [
              [0, 4, 0.0529582569989547],
              [5, 9, 0.0558095490543863],
              [10, 14, 0.0604326443303601],
              [15, 19, 0.0592331859478199],
              [20, 24, 0.0631016111246386],
              [25, 29, 0.0702732294593048],
              [30, 34, 0.0706992873192425],
              [35, 39, 0.0721768569258823],
              [40, 44, 0.065064037895203],
              [45, 49, 0.0622025507991608],
              [50, 54, 0.0589574456633972],
              [55, 59, 0.0616100529739052],
              [60, 64, 0.0655126196820712],
              [65, 69, 0.0604741357747057],
              [70, 74, 0.0503101544738328],
              [75, 100, 0.0711843815771348]
          ],
          "employment_rates_by_age": [
            [19,0.300],
            [20,0.693]
          ],
          "enrollment_rates_by_age": [
            [2,0],
            [3,0.529]
          ],
          "household_head_age_brackets": [
            [18,19],
            [20,24]
          ],
          "household_head_age_distribution_by_family_size": [
            [2,163,999],
            [3,115,757]
          ],
          "household_size_distribution": [
            [1,0.2781590909877753],
            [2,0.3443313103056699]
          ],
          "ltcf_resident_to_staff_ratio_distribution": [
            [1,1,0.0],
            [2,2,5.0]
          ],
          "ltcf_num_residents_distribution": [
            [0,19,0.0],
            [20,39,0.08955223880597014]
          ],
          "ltcf_num_staff_distribution": [
            [0,19,0.014925373134328358],
            [20,39,0.07462686567164178]
          ],
          "school_size_brackets": [
            [20,50],
            [51,100]
          ],
          "school_size_distribution": [
            0.027522935779816515,
            0.009174311926605505
          ],
          "school_size_distribution_by_type": [
            {
              "school_type": "pk-es",
              "size_distribution": [
                0.25,
                0.5
              ]
            },
            {
              "school_type": "ms",
              "size_distribution": [
                0.35,
                0.65
              ]
            }
          ],
          "school_types_by_age": [
            {
              "school_type": "pk-es",
              "age_range": [3,10]
            },
            {
              "school_type": "ms",
              "age_range": [11,13]
            }
          ],
          "workplace_size_counts_by_num_personnel": [
            [1,4,2947],
            [5,9,992]
          ]
        }"""
        return test_str

    def minimal_location_with_parent_test_str(self):
        test_str = """{
          "location_name": "test_location_child",
          "parent": {
              "location_name": "test_location_parent",
              "parent": {
                  "location_name": "test_location_grandparent",
                  "school_size_distribution_by_type": [
                    {
                      "school_type": "pk-es",
                      "size_distribution": [
                        0.25,
                        0.5
                      ]
                    },
                    {
                      "school_type": "ms",
                      "size_distribution": [
                        0.35,
                        0.65
                      ]
                    }
                  ],
                  "school_size_brackets": [
                    [20,50],
                    [51,100]
                  ],
                  "school_size_distribution": [
                    0.25,
                    0.75
                  ]
              },
              "population_age_distributions": [{
                  "num_bins": 16,
                  "distribution": [
                    [0, 4, 0.0529582569989547],
                    [5, 9, 0.0558095490543863],
                    [10, 14, 0.0604326443303601],
                    [15, 19, 0.0592331859478199],
                    [20, 24, 0.0631016111246386],
                    [25, 29, 0.0702732294593048],
                    [30, 34, 0.0706992873192425],
                    [35, 39, 0.0721768569258823],
                    [40, 44, 0.065064037895203],
                    [45, 49, 0.0622025507991608],
                    [50, 54, 0.0589574456633972],
                    [55, 59, 0.0616100529739052],
                    [60, 64, 0.0655126196820712],
                    [65, 69, 0.0604741357747057],
                    [70, 74, 0.0503101544738328],
                    [75, 100, 0.0711843815771348]
                  ]
                }]
          },
          "employment_rates_by_age": [
            [19,0.300],
            [20,0.693]
          ],
          "school_size_distribution": [
            0.45,
            0.55
          ]
        }"""
        return test_str

    def minimal_location_with_parent_filepath_test_str(self):
        test_str = """{
          "location_name": "test_location_child",
          "parent": "test_location_A.json",
          "employment_rates_by_age": [
            [19,0.300],
            [20,0.693]
          ],
          "school_size_distribution": [
            0.45,
            0.55
          ]
        }"""
        return test_str

    def test_load_completely_empty_object_test_str(self):
        """
        location_name is a required field, so a completely empty object should complain about missing that.
        """
        test_str = "{}"
        self.assertRaises(RuntimeError, sp.load_location_from_json_str, test_str)

    def test_load_empty_object_test_str(self):
        """
        Make sure that an empty json object populates all lists as empty.
        Because parts of the code rely on this assumption.
        location_name is a required field so it must be present.
        """
        test_str = """{"location_name": "test_location"}"""
        location = sp.load_location_from_json_str(test_str, check_constraints=False)
        self.assertEqual(location.location_name, "test_location",
                         "location_name incorrect")
        for list_property in location.get_list_properties():
            att = getattr(location, list_property)
            self.assertTrue(att is not None and len(att) == 0)

    def test_load_minimal_location(self):
        test_str = self.minimal_test_str()
        location = sp.load_location_from_json_str(test_str, check_constraints=False)

        self.assertEqual(location.location_name, "test_location",
                         "location_name incorrect")

        self.assertEqual(len(location.data_provenance_notices), 2,
                         "Array length incorrect")

        self.assertEqual(location.data_provenance_notices[0], "notice1",
                         "Array entry incorrect")

        self.assertEqual(location.data_provenance_notices[1], "notice2",
                         "Array entry incorrect")

        self.assertEqual(len(location.reference_links), 2,
                         "Array length incorrect")

        self.assertEqual(location.reference_links[0], "reference1",
                         "Array entry incorrect")

        self.assertEqual(location.reference_links[1], "reference2",
                         "Array entry incorrect")

        self.assertEqual(len(location.citations), 2,
                         "Array length incorrect")

        self.assertEqual(location.citations[0], "citation1",
                         "Array entry incorrect")
        self.assertEqual(location.citations[1], "citation2",
                         "Array entry incorrect")

        # Not checking parent field.

        self.assertEqual(len(location.population_age_distribution_16), 16,
                         "Array length incorrect")

        # Just checking the first couple entries
        self.assertEqual(len(location.population_age_distribution_16[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.population_age_distribution_16[0][0], 0,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distribution_16[0][1], 4,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distribution_16[0][2], 0.0529582569989547,
                         "Array entry incorrect")

        self.assertEqual(len(location.population_age_distribution_16[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.population_age_distribution_16[1][0], 5,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distribution_16[1][1], 9,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distribution_16[1][2], 0.0558095490543863,
                         "Array entry incorrect")

        self.assertEqual(len(location.employment_rates_by_age), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.employment_rates_by_age[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.employment_rates_by_age[0][0], 19,
                         "Array entry incorrect")
        self.assertEqual(location.employment_rates_by_age[0][1], 0.300,
                         "Array entry incorrect")

        self.assertEqual(len(location.employment_rates_by_age[1]), 2,
                         "Array length incorrect")
        self.assertEqual(location.employment_rates_by_age[1][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.employment_rates_by_age[1][1], 0.693,
                         "Array entry incorrect")

        self.assertEqual(len(location.enrollment_rates_by_age), 2,
                         "Array length incorrect")
        self.assertEqual(len(location.enrollment_rates_by_age[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.enrollment_rates_by_age[0][0], 2,
                         "Array entry incorrect")
        self.assertEqual(location.enrollment_rates_by_age[0][1], 0,
                         "Array entry incorrect")

        self.assertEqual(len(location.enrollment_rates_by_age[1]), 2,
                         "Array length incorrect")
        self.assertEqual(location.enrollment_rates_by_age[1][0], 3,
                         "Array entry incorrect")
        self.assertEqual(location.enrollment_rates_by_age[1][1], 0.529,
                         "Array entry incorrect")

        self.assertEqual(len(location.household_head_age_brackets), 2,
                         "Array length incorrect")
        self.assertEqual(len(location.household_head_age_brackets[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.household_head_age_brackets[0][0], 18,
                         "Array entry incorrect")
        self.assertEqual(location.household_head_age_brackets[0][1], 19,
                         "Array entry incorrect")

        self.assertEqual(len(location.household_head_age_brackets[1]), 2,
                         "Array length incorrect")
        self.assertEqual(location.household_head_age_brackets[1][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.household_head_age_brackets[1][1], 24,
                         "Array entry incorrect")

        self.assertEqual(len(location.household_head_age_distribution_by_family_size), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.household_head_age_distribution_by_family_size[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.household_head_age_distribution_by_family_size[0][0], 2,
                         "Array entry incorrect")
        self.assertEqual(location.household_head_age_distribution_by_family_size[0][1], 163,
                         "Array entry incorrect")
        self.assertEqual(location.household_head_age_distribution_by_family_size[0][2], 999,
                         "Array entry incorrect")

        self.assertEqual(len(location.household_head_age_distribution_by_family_size[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.household_head_age_distribution_by_family_size[1][0], 3,
                         "Array entry incorrect")
        self.assertEqual(location.household_head_age_distribution_by_family_size[1][1], 115,
                         "Array entry incorrect")
        self.assertEqual(location.household_head_age_distribution_by_family_size[1][2], 757,
                         "Array entry incorrect")

        self.assertEqual(len(location.household_size_distribution), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.household_size_distribution[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.household_size_distribution[0][0], 1,
                         "Array entry incorrect")
        self.assertEqual(location.household_size_distribution[0][1], 0.2781590909877753,
                         "Array entry incorrect")

        self.assertEqual(len(location.household_size_distribution[1]), 2,
                         "Array length incorrect")
        self.assertEqual(location.household_size_distribution[1][0], 2,
                         "Array entry incorrect")
        self.assertEqual(location.household_size_distribution[1][1], 0.3443313103056699,
                         "Array entry incorrect")

        self.assertEqual(len(location.ltcf_resident_to_staff_ratio_distribution), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.ltcf_resident_to_staff_ratio_distribution[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.ltcf_resident_to_staff_ratio_distribution[0][0], 1,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_resident_to_staff_ratio_distribution[0][1], 1,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_resident_to_staff_ratio_distribution[0][2], 0.0,
                         "Array entry incorrect")

        self.assertEqual(len(location.ltcf_resident_to_staff_ratio_distribution[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.ltcf_resident_to_staff_ratio_distribution[1][0], 2,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_resident_to_staff_ratio_distribution[1][1], 2,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_resident_to_staff_ratio_distribution[1][2], 5.0,
                         "Array entry incorrect")

        self.assertEqual(len(location.ltcf_num_residents_distribution), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.ltcf_num_residents_distribution[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.ltcf_num_residents_distribution[0][0], 0,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_residents_distribution[0][1], 19,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_residents_distribution[0][2], 0.0,
                         "Array entry incorrect")

        self.assertEqual(len(location.ltcf_num_residents_distribution[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.ltcf_num_residents_distribution[1][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_residents_distribution[1][1], 39,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_residents_distribution[1][2], 0.08955223880597014,
                         "Array entry incorrect")

        self.assertEqual(len(location.ltcf_num_staff_distribution), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.ltcf_num_staff_distribution[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.ltcf_num_staff_distribution[0][0], 0,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_staff_distribution[0][1], 19,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_staff_distribution[0][2], 0.014925373134328358,
                         "Array entry incorrect")

        self.assertEqual(len(location.ltcf_num_staff_distribution[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.ltcf_num_staff_distribution[1][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_staff_distribution[1][1], 39,
                         "Array entry incorrect")
        self.assertEqual(location.ltcf_num_staff_distribution[1][2], 0.07462686567164178,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_brackets), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.school_size_brackets[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_brackets[0][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_brackets[0][1], 50,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_brackets[1]), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_brackets[1][0], 51,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_brackets[1][1], 100,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_distribution), 2,
                         "Array length incorrect")

        self.assertEqual(location.school_size_distribution[0], 0.027522935779816515,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_distribution[1], 0.009174311926605505,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_distribution_by_type), 2,
                         "Array length incorrect")

        self.assertEqual(location.school_size_distribution_by_type[0].school_type, "pk-es",
                         "school_type incorrect")
        self.assertEqual(len(location.school_size_distribution_by_type[0].size_distribution), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_distribution_by_type[0].size_distribution[0], 0.25,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_distribution_by_type[0].size_distribution[1], 0.5,
                         "Array entry incorrect")

        self.assertEqual(location.school_size_distribution_by_type[1].school_type, "ms",
                         "school_type incorrect")
        self.assertEqual(len(location.school_size_distribution_by_type[1].size_distribution), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_distribution_by_type[1].size_distribution[0], 0.35,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_distribution_by_type[1].size_distribution[1], 0.65,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_types_by_age), 2,
                         "Array length incorrect")

        self.assertEqual(location.school_types_by_age[0].school_type, "pk-es",
                         "School type value incorrect")
        self.assertEqual(len(location.school_types_by_age[0].age_range), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_types_by_age[0].age_range[0], 3,
                         "Array entry incorrect")
        self.assertEqual(location.school_types_by_age[0].age_range[1], 10,
                         "Array entry incorrect")

        self.assertEqual(location.school_types_by_age[1].school_type, "ms",
                         "School type value incorrect")
        self.assertEqual(len(location.school_types_by_age[1].age_range), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_types_by_age[1].age_range[0], 11,
                         "Array entry incorrect")
        self.assertEqual(location.school_types_by_age[1].age_range[1], 13,
                         "Array entry incorrect")

        self.assertEqual(len(location.workplace_size_counts_by_num_personnel), 2,
                         "Array length incorrect")

        self.assertEqual(len(location.workplace_size_counts_by_num_personnel[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.workplace_size_counts_by_num_personnel[0][0], 1,
                         "Array entry incorrect")
        self.assertEqual(location.workplace_size_counts_by_num_personnel[0][1], 4,
                         "Array entry incorrect")
        self.assertEqual(location.workplace_size_counts_by_num_personnel[0][2], 2947,
                         "Array entry incorrect")

        self.assertEqual(len(location.workplace_size_counts_by_num_personnel[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.workplace_size_counts_by_num_personnel[1][0], 5,
                         "Array entry incorrect")
        self.assertEqual(location.workplace_size_counts_by_num_personnel[1][1], 9,
                         "Array entry incorrect")
        self.assertEqual(location.workplace_size_counts_by_num_personnel[1][2], 992,
                         "Array entry incorrect")

    def test_load_minimal_location_with_parent(self):
        test_str = self.minimal_location_with_parent_test_str()
        location = sp.load_location_from_json_str(test_str)
        self.check_minimal_location_with_parent(location)

    def test_load_minimal_location_with_parent_filepath(self):
        test_str = self.minimal_location_with_parent_filepath_test_str()
        location = sp.load_location_from_json_str(test_str, check_constraints=False)
        self.check_minimal_location_with_parent(location)

    def test_load_minimal_location_with_parent_filepath_from_filepath(self):
        child_filepath = "test_location_child.json"
        location = sp.load_location_from_filepath(child_filepath, check_constraints=False)
        self.check_minimal_location_with_parent(location)

    def check_minimal_location_with_parent(self, location):
        # All but the three specified lists are existing and empty...
        for list_property in location.get_list_properties():
            att = getattr(location, list_property)
            # what fields are we planning to test...
            if str(list_property) in ["employment_rates_by_age",
                                      "population_age_distributions",
                                      "school_size_distribution_by_type",
                                      "school_size_distribution",
                                      "school_size_brackets",
                                      "reference_links",
                                      "citations",
                                      "notes",
                                      ]:
                continue
            self.assertTrue(att is not None and len(att) == 0)  # everything else should be empty

        self.assertEqual(location.location_name, "test_location_child",
                         "location_name incorrect")

        self.assertEqual(len(location.employment_rates_by_age), 2,
                         "Array length incorrect")
        self.assertEqual(len(location.employment_rates_by_age[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.employment_rates_by_age[0][0], 19,
                         "Array entry incorrect")
        self.assertEqual(location.employment_rates_by_age[0][1], 0.3,
                         "Array entry incorrect")
        self.assertEqual(len(location.employment_rates_by_age[1]), 2,
                         "Array length incorrect")
        self.assertEqual(location.employment_rates_by_age[1][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.employment_rates_by_age[1][1], 0.693,
                         "Array entry incorrect")

        self.assertEqual(location.population_age_distributions[0].num_bins, 16,
                         "Num bins incorrect")
        self.assertEqual(len(location.population_age_distributions[0].distribution), 16,
                         "Array length incorrect")

        # Just checking the first couple entries.
        self.assertEqual(len(location.population_age_distributions[0].distribution[0]), 3,
                         "Array length incorrect")
        self.assertEqual(location.population_age_distributions[0].distribution[0][0], 0,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distributions[0].distribution[0][1], 4,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distributions[0].distribution[0][2], 0.0529582569989547,
                         "Array entry incorrect")
        self.assertEqual(len(location.population_age_distributions[0].distribution[1]), 3,
                         "Array length incorrect")
        self.assertEqual(location.population_age_distributions[0].distribution[1][0], 5,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distributions[0].distribution[1][1], 9,
                         "Array entry incorrect")
        self.assertEqual(location.population_age_distributions[0].distribution[1][2], 0.0558095490543863,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_distribution_by_type), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_distribution_by_type[0].school_type, "pk-es",
                         "school_type incorrect")
        self.assertEqual(len(location.school_size_distribution_by_type[0].size_distribution), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_distribution_by_type[0].size_distribution[0], 0.25,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_distribution_by_type[0].size_distribution[1], 0.5,
                         "Array entry incorrect")

        self.assertEqual(location.school_size_distribution_by_type[1].school_type, "ms",
                         "school_type incorrect")
        self.assertEqual(len(location.school_size_distribution_by_type[1].size_distribution), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_distribution_by_type[1].size_distribution[0], 0.35,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_distribution_by_type[1].size_distribution[1], 0.65,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_brackets), 2,
                         "Array length incorrect")
        self.assertEqual(len(location.school_size_brackets[0]), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_brackets[0][0], 20,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_brackets[0][1], 50,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_brackets[1][0], 51,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_brackets[1][1], 100,
                         "Array entry incorrect")

        self.assertEqual(len(location.school_size_distribution), 2,
                         "Array length incorrect")
        self.assertEqual(location.school_size_distribution[0], 0.45,
                         "Array entry incorrect")
        self.assertEqual(location.school_size_distribution[1], 0.55,
                         "Array entry  incorrect")

    @unittest.skip("constraint check not working properly, need investigation.")
    def test_constraint_check(self):
        location = sp.load_location_from_filepath("test_location_grand_child.json", check_constraints=True)
        self.assertEqual(location['school_size_distribution'], [0.5, 0.5])

        with self.assertWarns(Warning) as wn:
            sp.load_location_from_filepath("test_location_bad.json", check_constraints=True)
            self.assertTrue('has some negative values' in str(wn), 'Check failed: expect to get negative distribution check messages')


class TestChecks(unittest.TestCase):
    """
    Test checks can be run on probability distributions. Checks made: sum of
    probability distributions is close to 1, distribution has no negative values.
    """
    def test_check_probability_distribution_sums(self, location_name='usa-Washington-seattle_metro', property_list=None, tolerance=1e-2):
        """
        Run all checks for fields in property_list representing probability distributions. Each
        should have a sum that equals 1 within the tolerance level.

        Args:
            location_name(str)   : name of the location json to test
            property_list (list) : list of properties to check the sum of the probabilityd distribution
            tolerance (float)    : difference from the sum of 1 tolerated
        """
        location_file_path = f"{location_name}.json"
        location = sp.load_location_from_filepath(location_file_path)

        if property_list is None:
            sp.logger.info(f"\nTesting all probability distributions sum to 1 or within tolerance {tolerance} for {location_name}.")
            checks, msgs = sp.check_all_probability_distribution_sums(location, tolerance)

            err_msgs = [msg for msg in msgs if msg is not None]  # only get the msgs for failures
            err_msg = "\n".join(err_msgs)
            assert sum(checks) == len(checks), err_msg  # assert that all checks passed
            print(f'All {sum(checks)} checks passed.')

        else:
            # Example of how the sum checks can be run for a subset of properties
            sp.logger.info(f"\nTesting a subset of probability distributions sum to 1 or within tolerance {tolerance} for {location_name}.")
            for i, property_name in enumerate(property_list):
                check, msg = sp.check_probability_distribution_sum(location, property_name, tolerance)
                assert check == True, msg
                print(f'{property_name} check passed.')

    def test_check_probability_distribution_nonnegative(self, location_name='usa-Washington-seattle_metro', property_list=None):
        """
        Run all checks for fields in property_list representing probability distributions. Each
        should have all non negative values.

        Args:
            location_name(str)   : name of the location json to test
            property_list (list) : list of properties to check the sum of the probabilityd distribution
        """
        location_file_path = f"{location_name}.json"
        location = sp.load_location_from_filepath(location_file_path)

        if property_list is None:
            sp.logger.info(f"\nTesting all probability distributions are all non negative for {location_name}.")
            checks, msgs = sp.check_all_probability_distribution_nonnegative(location)

            err_msgs = [msg for msg in msgs if msg is not None]
            err_msg = "\n".join(err_msgs)
            assert sum(checks) == len(checks), err_msg  # assert that all checks passed
            print(f'All {sum(checks)} checks passed.')

        else:
            # Examples of how the non negative checks can be run for a subset of properties
            sp.logger.info(f"\nTesting a subset of probability distributions are all non negative for {location_name}")
            for i, property_name in enumerate(property_list):
                check, msg = sp.check_probability_distribution_nonnegative(location, property_name)
                assert check == True, msg
                print(f'{property_name} check passed.')


class Testconvert_df_to_json_array(unittest.TestCase):
    """
    Test different aspects of the sp.data.convert_df_to_json_array() method.
    """
    pars = sc.objdict(
        location_name='usa-Washington',  # name of the location
        property_name='population_age_distributions',  # name of the property to compare to
        cols_ind=[],  # list of column indices to include in array in conversion
        int_cols_ind=[],  # list of column induces to convert to ints
    )
    @classmethod
    def setUpClass(cls) -> None:
        # for Testconvert_df_to_json_array()
        cls.pars['filepath'] = os.path.join(sc.thisdir(__file__, 'testdata'), 'Washington_age_bracket_distr_16.dat')

    def setup_convert_df_to_json_array(self, pars):
        """
        Set up objects to compare.

        Args:
            pars (dict): dictionary to get the data array and json array for comparison.

        Returns:
            array, json.array : An array of the desired data from a dataframe and
            the json entry for comparison.
        """
        df = pd.read_csv(pars.filepath)

        # columns to include : include all by default
        if pars.cols_ind == []:
            cols = df.columns
        else:
            cols = df.columns[pars.cols_ind]  # use indices to indicate which columns to include

        if pars.int_cols_ind == []:
            int_cols = pars.int_cols_ind
        else:
            int_cols = list(df.columns[pars.int_cols_ind].values)

        # array-ify all the data, convert some columns to integers
        arr = sp.convert_df_to_json_array(df, cols, int_cols)

        # corresponding json data object for the same location and data
        location = sp.load_location_from_filepath(f"{pars.location_name}.json")

        json_array = getattr(location, pars.property_name)

        if pars.property_name == 'population_age_distributions':
            json_array = [j for j in json_array if j.num_bins == len(arr)][0].distribution

        return arr, json_array

    def test_convert_df_to_json_array_age_distribution_16(self, verbose=False):
        """
        Test that the sp.convert_df_to_json_entry() converts the desired data from
        a dataframe to an array of arrays like those that can be uploaded to the
        json data objects synthpops uses.
        """
        sp.logger.info("Testing method to convert pandas dataframe to json arrays.")

        arr, json_array = self.setup_convert_df_to_json_array(self.pars)
        assert arr == json_array, "Arrays don't match"

        if verbose:
            print(f"The pandas table converted to an array matches the corresponding json array for {pars.property_name} in location: {pars.location_name}")

    def test_convert_df_to_json_entry_int_values(self):
        """
        Test that when converting a df to arrays, some of the columns specified are
        made into ints, like when we have columns specifying the age bracket min or
        max values.
        """
        sp.logger.info("Test that specified columns are converted to ints when data from a df is converted to a json array.")
        test_pars = sc.dcp(self.pars)
        test_pars.int_cols_ind = [0, 1]  # want to convert values from columns 0 and 1 to integers

        arr, json_array = self.setup_convert_df_to_json_array(test_pars)

        for i in range(len(arr)):
            for j in test_pars.int_cols_ind:
                assert isinstance(arr[i][j], int), f"Value at ({i},{j}): {arr[i][j]} is not an integer as expected."
        print("Check passed. Some columns were converted to ints as expected.")


if __name__ == '__main__':

    unittest.main(verbosity=2)  # run all tests in this file

    # # run tests with non default values
    # t1 = TestLocation()
    # t1.test_load_completely_empty_object_test_str()
    # t1.test_load_empty_object_test_str()
    # t1.test_load_minimal_location()
    # t1.test_load_minimal_location_with_parent()
    # t1.test_load_minimal_location_with_parent_filepath()
    # t1.test_load_minimal_location_with_parent_filepath_from_filepath()

    # # run checks on a subset of the properties by specifying property_list
    # t2 = TestChecks()
    # t2.test_check_probability_distribution_sums(property_list=['population_age_distributions', 'household_size_distribution'])
    # t2.test_check_probability_distribution_nonnegative(property_list=['household_size_distribution'])

    # t3 = Testconvert_df_to_json_array()
    # t3.test_convert_df_to_json_array_age_distribution_16()
    # t3.test_convert_df_to_json_entry_int_values()
