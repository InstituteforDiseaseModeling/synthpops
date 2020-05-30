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
        self.d_datadir = sp.datadir
        self.d_location = "seattle_metro"
        self.d_state_location = "Washington"
        self.d_country_location = "usa"
        self.d_sheet_name="United States of America"
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

    def test_contact_matrix_has_all_layers(self):
        contact_matrix = sp.get_contact_matrix_dic(
            datadir=sp.datadir,
            sheet_name="United States of America"
        )
        for layer in ['H','S','W','C']:
            self.assertIn(layer, contact_matrix)
        pass

    def get_seattle_household_size_distro(self):
        hh_distro = sp.get_household_size_distr(
            datadir=self.d_datadir,
            location=self.d_location,
            state_location=self.d_state_location,
            country_location=self.d_country_location
        )
        return hh_distro

    def verify_portion_honored(self, probability_buckets, count_buckets, portion=0.5):
        if self.is_debugging:
            print(f"prob buckets: {probability_buckets}")
            print(f"count buckets: {count_buckets}")
        num_portions = 1.0 / portion
        curr_portion = 1
        excess_probability = 0
        excess_count = 0
        sum_count = sum(count_buckets)
        count_portion = int(sum_count * portion)

        prob_indexes = []
        count_indexes = []

        bucket_index = 0
        count_index = 0
        while curr_portion < num_portions + 1:
            total_probability = excess_probability
            # for loop test first until portion * num_portions == 1.0
            while total_probability < portion and bucket_index < len(probability_buckets): # Get bucket index where total prob > portion
                total_probability += probability_buckets[bucket_index]
                bucket_index += 1
            prob_indexes.append(bucket_index)
            excess_probability = total_probability - portion

            total_counts = excess_count
            while total_counts < count_portion and count_index < len(count_buckets):
                total_counts += count_buckets[count_index]
                count_index += 1
            count_indexes.append(count_index)
            excess_count = total_counts - count_portion
            curr_portion += 1
            if self.is_debugging:
                print(f"prob_indexes: {prob_indexes}")
                print(f"count_indexes: {count_indexes}")

        if self.is_debugging:
            print(f"prob_indexes: {prob_indexes}")
            print(f"count_indexes: {count_indexes}")
        for x in range (len(prob_indexes)):
            index_diff = abs(prob_indexes[x] - count_indexes[x])
            self.assertLessEqual(index_diff, 1,
                                 msg="indexes shouldn't be off by more than one. "
                                     f"probability_indexes: {prob_indexes} "
                                     f"count_indexes: {count_indexes}")

    def test_seattle_household_size_distro_honored(self):
        self.is_debugging = False
        hh_distro = self.get_seattle_household_size_distro()
        hh_sizes = sp.generate_household_sizes(500, hh_distro)

        hh_size_list = list(hh_sizes)  # Comes as np.ndarray
        fewest_houses = min(hh_size_list)
        fewest_index = hh_size_list.index(fewest_houses)

        most_houses = max(hh_size_list)
        most_index = hh_size_list.index(most_houses)

        highest_probability = max(hh_distro.values())
        lowest_probability = min(hh_distro.values())

        most_houses_probability = hh_distro[most_index + 1] # hh_distro is 1 indexed
        fewest_houses_probability = hh_distro[fewest_index + 1]

        self.assertEqual(highest_probability, most_houses_probability,
                         msg="The most common household size should be the size with the highest probability")

        prob_bucket_list = list(hh_distro.values())
        self.verify_portion_honored(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list,
            portion=0.25
        )
        self.verify_portion_honored(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list,
            portion=0.2
        )
        self.verify_portion_honored(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list,
            portion=0.1
        )

    def test_custom_household_size_distro_honored(self):
        self.is_debugging = False
        custom_distro = {
            1: 0.25,
            2: 0.075,
            3: 0.10,
            4: 0.30,
            5: 0.05,
            6: 0.05,
            7: 0.175
        }
        hh_sizes = sp.generate_household_sizes(500, custom_distro)

        hh_size_list = list(hh_sizes)  # Comes as np.ndarray
        fewest_houses = min(hh_size_list)
        fewest_index = hh_size_list.index(fewest_houses)

        most_houses = max(hh_size_list)
        most_index = hh_size_list.index(most_houses)

        highest_probability = max(custom_distro.values())
        lowest_probability = min(custom_distro.values())

        most_houses_probability = custom_distro[most_index + 1] # hh_distro is 1 indexed
        fewest_houses_probability = custom_distro[fewest_index + 1]

        self.assertEqual(highest_probability, most_houses_probability,
                         msg="The most common household size should be the size with the highest probability")

        prob_bucket_list = list(custom_distro.values())
        self.verify_portion_honored(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list,
            portion=0.25
        )
        self.verify_portion_honored(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list,
            portion=0.2
        )
        self.verify_portion_honored(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list,
            portion=0.1
        )

    def test_household_size_distribution_adds_up(self):
        hh_distro = self.get_seattle_household_size_distro()

        total_sizes = sum(hh_distro.values())
        self.assertNotIn(0, hh_distro,
                         msg="Each key is a household size, 0 shouldn't be here")
        for x in range(1, 8):
            self.assertIn(x, hh_distro,
                          msg=f"Households come in 1 to 7. Size {x} should be in here.")
        self.assertEqual(total_sizes, 1,
                         msg=f"This is a probability distribution that should add up to 1. Got "
                             f"{total_sizes} from this: {hh_distro}")
        if self.is_debugging:
            print(total_sizes)
            print(hh_distro)

if __name__ == "__main__":
    test = HouseholdsTest()
    test.setUp()
    test.test_seattle_age_distro_honored()