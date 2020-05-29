import unittest
from copy import deepcopy
import synthpops as sp

seapop_500 = sp.generate_synthetic_population(
    n=500,
    datadir=sp.datadir,
    location='seattle_metro',
    state_location='Washington',
    country_location='usa',
    sheet_name='United States of America',
    school_enrollment_counts_available=False,
    verbose=False,
    plot=False,
    write=True,
    return_popdict=True,
    use_default=False
)

import json


class HouseholdsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.is_debugging = False
        pass

    def tearDown(self) -> None:
        pass

    def remove_sets_from_contacts(self, person):
        trim_person = {}
        for k in person:
            if k != 'contacts':
                trim_person[k] = person[k]
        return trim_person

    def test_seapop_500_every_human_one_household(self):
        my_seapop_500 = deepcopy(seapop_500)
        households = {}
        hh_index = 0
        current_household_list = None
        current_household_list_index = 0
        current_person_index = 0
        current_human = None
        while current_person_index < 500:
            if not current_household_list:
                current_reference_person = my_seapop_500[current_person_index]
                current_household_list = list(current_reference_person['contacts']['H'])
                current_household_list_index = 0
                current_human = current_reference_person
                households[hh_index] = []
                households[hh_index].append(self.remove_sets_from_contacts(current_human))
                current_person_index += 1
            else:
                for p in current_household_list:
                    current_human = my_seapop_500[current_household_list[current_household_list_index]]
                    households[hh_index].append(self.remove_sets_from_contacts(current_human))
                    current_household_list_index += 1
                    current_person_index += 1
                current_household_list = None
                hh_index += 1
        if self.is_debugging:
            with open ("DEBUG_sea500_household.json","w") as outfile:
                json.dump(households, outfile, indent=4, sort_keys=True)

        total_households = len(households)
        sizes = []
        for h in households:
            this_household = households[h]
            sizes.append(len(this_household))
            household_id = this_household[0]['hhid']
            for j in this_household:
                self.assertEqual(j['hhid'], household_id,
                                 msg=f"Expect all members of household {h} to have "
                                     f"household id {household_id}. Person {j} has "
                                     f"{j['hhid']}")
        if self.is_debugging:
            print(f"Total households: {total_households}")
            print(f"Mean household size: {sum(sizes) / total_households}")
        self.assertEqual(sum(sizes), len(my_seapop_500),
                         msg=f"Each person should be in one household. Total household pop: {sum(sizes)}, "
                             f"Population size: {len(my_seapop_500)}.")

    def verify_age_bracket_dictionary_correct(self, age_by_brackets_dic):
        if self.is_debugging:
            age_bb_json = {}
            for k in age_by_brackets_dic:
                age_bb_json[int(k)] = age_by_brackets_dic[k]
            with open(f"DEBUG_{self._testMethodName}_age_dict.json","w") as outfile:
                json.dump(age_bb_json, outfile, indent=4, sort_keys=True)

        max_year = 100
        ages = range(0, max_year)
        expected_bucket = 0
        previous_age = None

        # # Induce error for testing the test
        # age_by_brackets_dic[30] = 6
        # age_by_brackets_dic[31] = 8

        for age in ages:
            bucket = age_by_brackets_dic[age]
            self.assertEqual(type(bucket), int,
                             msg=f"Each age should have a single bucket id (int). got: {bucket}")
            if bucket > expected_bucket:
                self.assertEqual(bucket, expected_bucket + 1,
                                 msg=f"Buckets should increase by 1 only. At previous age: {previous_age} "
                                     f"got bucket: {expected_bucket}. At {age} got {bucket}.")
                expected_bucket += 1
            else:
                self.assertEqual(bucket, expected_bucket,
                                 msg=f"Buckets should increase by 1 only. At previous age: {previous_age} "
                                     f"got bucket: {expected_bucket}. At {age} got {bucket}.")
            previous_age = age
        self.assertLess(expected_bucket, max_year, msg=f"There should be less buckets than ages. Got "
                                                   f"{expected_bucket} for {max_year} ages.")
        pass

    def test_seattle_age_brackets(self):
        self.is_debugging = False
        age_brackets = sp.get_census_age_brackets(
            datadir=sp.datadir,
            state_location="Washington",
            country_location="usa",
            use_default=False
        )
        age_brackets_json = {}
        for k in age_brackets:
            age_brackets_json[k] = age_brackets[k].tolist()
        if self.is_debugging:
            with open(f"DEBUG_{self._testMethodName}_age_brackets.json","w") as outfile:
                json.dump(age_brackets_json, outfile, indent=4)
        age_by_brackets_dic = sp.get_age_by_brackets_dic(
            age_brackets=age_brackets
        )
        self.verify_age_bracket_dictionary_correct(age_by_brackets_dic)

    def test_custom_age_brackets(self):
        self.is_debugging = True
        college_years = list(range(19, 23))
        early_career = list(range(23, 30))
        mid_career = list(range(30, 50))
        late_career = list(range(50, 65))
        retirement = list(range(65, 80))
        managed_care = list(range(80, 100))
        my_age_brackets = {
            0: [0, 1],
            1: [2, 3, 4],
            2: [5, 6, 7, 8, 9, 10, 11],
            3: [12, 13, 14],
            4: [15, 16, 17, 18],
            5: college_years,
            6: early_career,
            7: mid_career,
            8: late_career,
            9: retirement,
            10: managed_care
        }
        age_by_brackets_dic = sp.get_age_by_brackets_dic(
            age_brackets=my_age_brackets
        )

        self.verify_age_bracket_dictionary_correct(
            age_by_brackets_dic=age_by_brackets_dic
        )
        pass

    def test_seattle_age_distro_honored(self):


