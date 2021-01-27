import synthpops as sp

import unittest

class TestLocation(unittest.TestCase):
    """
    These tests need to be run from the root synthpops folder, because some of the tests involve relative
    filepath assumptions based on that.
    """

    def minimal_test_str(self):
        test_str = """{
          "location_name": "test_location",
          "data_provenance_notices": ["notice1","notice2"],
          "reference_links": ["reference1","reference2"],
          "citations": ["citation1","citation2"],
          "parent": {},
          "population_age_distribution": [
            [0,4,0.06],
            [5,9,0.20]
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
                0.6
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
                        0.6
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
              "population_age_distribution": [
                [0,4,0.06],
                [5,9,0.20]
              ]
          },
          "employment_rates_by_age": [
            [19,0.300],
            [20,0.693]
          ],
          "school_size_distribution": [
            0.45,
            0.65
          ]
        }"""
        return test_str

    def minimal_location_with_parent_filepath_test_str(self):
        test_str = """{
          "location_name": "test_location_child",
          "parent": "tests/unittests/test_location_A.json",
          "employment_rates_by_age": [
            [19,0.300],
            [20,0.693]
          ],
          "school_size_distribution": [
            0.45,
            0.65
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
        location = sp.load_location_from_json_str(test_str)
        self.assertEqual(location.location_name, "test_location",
                          "location_name incorrect")
        for list_property in location.get_list_properties():
            att = getattr(location, list_property)
            self.assertTrue(att is not None and len(att) == 0)

    def test_load_minimal_location(self):
        test_str = self.minimal_test_str()
        location = sp.load_location_from_json_str(test_str)

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

        self.assertEqual(len(location.population_age_distribution), 2,
                          "Array length incorrect")

        self.assertEqual(len(location.population_age_distribution[0]), 3,
                          "Array length incorrect")
        self.assertEqual(location.population_age_distribution[0][0], 0,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[0][1], 4,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[0][2], 0.06,
                          "Array entry incorrect")

        self.assertEqual(len(location.population_age_distribution[1]), 3,
                          "Array length incorrect")
        self.assertEqual(location.population_age_distribution[1][0], 5,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[1][1], 9,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[1][2], 0.20,
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
        self.assertEqual(location.school_size_distribution_by_type[1].size_distribution[1], 0.6,
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
        location = sp.load_location_from_json_str(test_str)
        self.check_minimal_location_with_parent(location)

    def test_load_minimal_location_with_parent_filepath_from_filepath(self):
        child_filepath = "tests/unittests/test_location_child.json"
        location = sp.load_location_from_filepath(child_filepath)
        self.check_minimal_location_with_parent(location)

    def check_minimal_location_with_parent(self, location):
        # All but the three specified lists are existing and empty...
        for list_property in location.get_list_properties():
            att = getattr(location, list_property)
            if str(list_property) in ["employment_rates_by_age",
                                      "population_age_distribution",
                                      "school_size_distribution_by_type",
                                      "school_size_distribution",
                                      "school_size_brackets",
                                      ]:
                continue
            self.assertTrue(att is not None and len(att) == 0)

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

        self.assertEqual(len(location.population_age_distribution), 2,
                          "Array length incorrect")

        self.assertEqual(len(location.population_age_distribution[0]), 3,
                          "Array length incorrect")
        self.assertEqual(location.population_age_distribution[0][0], 0,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[0][1], 4,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[0][2], 0.06,
                          "Array entry incorrect")
        self.assertEqual(len(location.population_age_distribution[1]), 3,
                          "Array length incorrect")
        self.assertEqual(location.population_age_distribution[1][0], 5,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[1][1], 9,
                          "Array entry incorrect")
        self.assertEqual(location.population_age_distribution[1][2], 0.2,
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
        self.assertEqual(location.school_size_distribution_by_type[1].size_distribution[1], 0.6,
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
        self.assertEqual(location.school_size_distribution[1], 0.65,
                          "Array entry  incorrect")
