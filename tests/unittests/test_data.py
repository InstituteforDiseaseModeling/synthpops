import synthpops as sp

import unittest

class TestLocation(unittest.TestCase):

    def nominal_test_str(self):
        test_str = """{
          "data_provenance_notices": ["notice1","notice2"],
          "reference_links": ["reference1","reference2"],
          "citations": ["citation1","citation2"],
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
          ]
        }"""
        return test_str

    def test_load_location(self):
        test_str = self.nominal_test_str()
        location = sp.load_location_from_json_str(test_str)

        self.assertEquals(len(location.data_provenance_notices), 2,
                          "Array length incorrect")

        self.assertEquals(location.data_provenance_notices[0], "notice1",
                          "Array entry incorrect")

        self.assertEquals(location.data_provenance_notices[1], "notice2",
                          "Array entry incorrect")

        self.assertEquals(len(location.reference_links), 2,
                          "Array length incorrect")

        self.assertEquals(location.reference_links[0], "reference1",
                          "Array entry incorrect")

        self.assertEquals(location.reference_links[1], "reference2",
                          "Array entry incorrect")

        self.assertEquals(len(location.citations), 2,
                          "Array length incorrect")

        self.assertEquals(location.citations[0], "citation1",
                          "Array entry incorrect")
        self.assertEquals(location.citations[1], "citation2",
                          "Array entry incorrect")

        self.assertEquals(len(location.population_age_distribution), 2,
                          "Array length incorrect")

        self.assertEquals(len(location.population_age_distribution[0]), 3,
                          "Array length incorrect")
        self.assertEquals(location.population_age_distribution[0][0], 0,
                          "Array entry incorrect")
        self.assertEquals(location.population_age_distribution[0][1], 4,
                          "Array entry incorrect")
        self.assertEquals(location.population_age_distribution[0][2], 0.06,
                          "Array entry incorrect")

        self.assertEquals(len(location.population_age_distribution[1]), 3,
                          "Array length incorrect")
        self.assertEquals(location.population_age_distribution[1][0], 5,
                          "Array entry incorrect")
        self.assertEquals(location.population_age_distribution[1][1], 9,
                          "Array entry incorrect")
        self.assertEquals(location.population_age_distribution[1][2], 0.20,
                          "Array entry incorrect")

        self.assertEquals(len(location.employment_rates_by_age), 2,
                          "Array length incorrect")

        self.assertEquals(len(location.employment_rates_by_age[0]), 2,
                          "Array length incorrect")
        self.assertEquals(location.employment_rates_by_age[0][0], 19,
                          "Array entry incorrect")
        self.assertEquals(location.employment_rates_by_age[0][1], 0.300,
                          "Array entry incorrect")

        self.assertEquals(len(location.employment_rates_by_age[1]), 2,
                          "Array length incorrect")
        self.assertEquals(location.employment_rates_by_age[1][0], 20,
                          "Array entry incorrect")
        self.assertEquals(location.employment_rates_by_age[1][1], 0.693,
                          "Array entry incorrect")

        self.assertEquals(len(location.enrollment_rates_by_age), 2,
                          "Array length incorrect")
        self.assertEquals(len(location.enrollment_rates_by_age[0]), 2,
                          "Array length incorrect")
        self.assertEquals(location.enrollment_rates_by_age[0][0], 2,
                          "Array entry incorrect")
        self.assertEquals(location.enrollment_rates_by_age[0][1], 0,
                          "Array entry incorrect")

        self.assertEquals(len(location.enrollment_rates_by_age[1]), 2,
                          "Array length incorrect")
        self.assertEquals(location.enrollment_rates_by_age[1][0], 3,
                          "Array entry incorrect")
        self.assertEquals(location.enrollment_rates_by_age[1][1], 0.529,
                          "Array entry incorrect")

