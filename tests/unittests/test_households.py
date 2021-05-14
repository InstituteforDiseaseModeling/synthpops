"""
This file includes tests for the household settings,
When investigation is needed, set the self.is_debugging = True

"""
import pytest
import unittest
import numpy as np
import json
import collections
import scipy
import utilities
import synthpops as sp
from synthpops import households as sphh
from synthpops import data_distributions as spdd

# the default test data was generated for 500 people using the below parameters
# and each test case will validate the properties of the population named "seapop_500"
seapop_500 = sp.generate_synthetic_population(
    n=500,
    datadir=sp.settings.datadir,
    location='seattle_metro',
    state_location='Washington',
    country_location='usa',
    sheet_name='United States of America',
    plot=False,
    write=False,
    return_popdict=True,
    use_default=False,
)


class HouseholdsTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up class variables

        Returns:
            None
        """
        np.random.seed(0)
        self.is_debugging = False
        self.d_datadir = sp.settings.datadir
        self.d_location = "seattle_metro"
        self.d_state_location = "Washington"
        self.d_country_location = "usa"
        self.d_sheet_name = "United States of America"
        pass

    def tearDown(self) -> None:
        pass

    def remove_sets_from_contacts(self, person):
        """
        Helper method to remove contact attribute from the person dictionary.

        Args:
            person: a person (item) of the pop dictionary

        Returns:
            A person (item) stripped of contacts attribute.
        """
        trim_person = {}
        for k in person:
            if k != 'contacts':
                trim_person[k] = person[k]
        return trim_person

    def test_seapop_500_every_human_one_household(self):
        """
        Loop over the target population and check the household layer and make sure

        (1) All members of household has the correct hhid
        (2) Each person should be in one household

        If is_debugging the true, population will be saved to
        DEBUG_sea500_household.json.

        Returns:
            None
        """
        self.is_debugging = False
        # my_seapop_500 = deepcopy(seapop_500)
        my_seapop_500 = seapop_500
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
                if len(current_household_list) == 0:
                    hh_index += 1
                    current_household_list = None
            else:
                for p in current_household_list:
                    current_human = my_seapop_500[current_household_list[current_household_list_index]]
                    households[hh_index].append(self.remove_sets_from_contacts(current_human))
                    current_household_list_index += 1
                    current_person_index += 1
                current_household_list = None
                hh_index += 1
        if self.is_debugging:
            with open("DEBUG_sea500_household.json", "w") as outfile:
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

    def verify_age_bracket_dictionary_correct(self, age_by_brackets):
        """
        Validation method for the result from get_age_by_brackets including:

        (1) Each age should have a single bucket index
        (2) Buckets index increment is 1
        (3) There should be fewer buckets than ages

        Args:
            age_by_brackets_dic: age by brackets dictionary for lookup

        Returns:
            None
        """
        if self.is_debugging:
            age_bb_json = {}
            for k in age_by_brackets:
                age_bb_json[int(k)] = age_by_brackets[k]
            with open(f"DEBUG_{self._testMethodName}_age_dict.json", "w") as outfile:
                json.dump(age_bb_json, outfile, indent=4, sort_keys=True)

        max_year = 100
        ages = range(0, max_year)
        expected_bucket = 0
        previous_age = None

        # # Induce error for testing the test
        # age_by_brackets[30] = 6
        # age_by_brackets[31] = 8

        for age in ages:
            bucket = age_by_brackets[age]
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
        """
        Test for method get_census_age_brackets and get_age_by_brackets. It
        calls helper method verify_age_bracket_dictionary_correct for
        verification.

        Returns:
            None
        """
        self.is_debugging = False
        age_brackets = spdd.get_census_age_brackets(
            datadir=sp.settings.datadir,
            state_location="Washington",
            country_location="usa",
            use_default=False
        )
        age_brackets_json = {}
        for k in age_brackets:
            age_brackets_json[k] = age_brackets[k].tolist()
        if self.is_debugging:
            with open(f"DEBUG_{self._testMethodName}_age_brackets.json", "w") as outfile:
                json.dump(age_brackets_json, outfile, indent=4)
        age_by_brackets = sp.get_age_by_brackets(
            age_brackets=age_brackets
        )
        self.verify_age_bracket_dictionary_correct(age_by_brackets)

    def test_custom_age_brackets(self):
        """
        Use custom age_brackets to make sure method get_age_by_brackets
        behaves correctly. The validation logic is in
        verify_age_bracket_dictionary_correct method.

        Returns:
            None
        """
        self.is_debugging = False
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
        age_by_brackets = sp.get_age_by_brackets(
            age_brackets=my_age_brackets
        )

        self.verify_age_bracket_dictionary_correct(
            age_by_brackets=age_by_brackets
        )
        pass

    def test_contact_matrix_has_all_layers(self):
        """
        Test get_contact_matrices method to make sure it contains all layers
        'H', 'S', 'W', 'C'.

        Returns:
            None
        """
        contact_matrix = sp.get_contact_matrices(
            datadir=sp.settings.datadir,
            sheet_name="United States of America"
        )
        for layer in ['H', 'S', 'W', 'C']:
            self.assertIn(layer, contact_matrix)
        pass

    def get_seattle_household_size_distro(self):
        """
        Helper method to test get_household_size_distr.

        Returns:
            Household size distribution obtained from get_household_size_distr
            method.
        """
        hh_distro = spdd.get_household_size_distr(
            datadir=self.d_datadir,
            location=self.d_location,
            state_location=self.d_state_location,
            country_location=self.d_country_location
        )
        return hh_distro

    def verify_buckets(self, probability_buckets, count_buckets):
        """
        This method use chi-square statistic test to make sure the actual data
        matches the expected probability.

        Args:
            probability_buckets (np.ndarray) : array of expected probablity, must sum to 1
            count_buckets (np.ndarray)       : actual count

        Returns:
            None
        """
        expected_bucket = [i * sum(count_buckets) for i in probability_buckets]
        utilities.statistic_test(expected=expected_bucket, actual=count_buckets, test="x")

    def verify_portion_honored(self, probability_buckets, count_buckets, portion=0.5):
        """
        This was an old verification written by cwiswell which checks if the
        actual probablity falls within the expected portions (with some error
        tolerated). For example, if the probabilities expected are [0.2, 0.3,
        0.1, 0.1, 0.3] and portion = 0.2 then the space was split to 5 equally
        spaced portions and if the bucket has 100 items in total, we would
        expect to see [20, 30, 10, 10, 30], this method checks if the actual
        count is off from the expectation.

        Args:
            probability_buckets (np.ndarray) : expected probablity
            count_buckets (np.ndarray)       : actual count
            portion (float)                  : use to split the space, for example if portion=0.25, the space was split to 4 quartiles

        Returns:
            None
        """
        num_portions = 1.0 / portion
        curr_portion = 1
        excess_probability = 0
        excess_count = 0
        sum_count = sum(count_buckets)
        count_portion = int(sum_count * portion)
        if self.is_debugging:
            print(f"prob buckets: {probability_buckets}")
            print(f"count buckets: {count_buckets}")
            print(f"sum counts: {sum_count}")
            print(f"count_portion: {count_portion}")

        prob_indexes = []
        count_indexes = []

        bucket_index = 0
        count_index = 0
        while curr_portion < num_portions + 1:
            total_probability = excess_probability
            # for loop test first until portion * num_portions == 1.0
            while total_probability < portion and bucket_index < len(
                    probability_buckets):  # Get bucket index where total prob > portion
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
        max_diff = num_portions / 5
        if max_diff < 1:
            max_diff = 1
        total_diff = 0
        for x in range(len(prob_indexes)):
            local_diff = prob_indexes[x] - count_indexes[x]
            total_diff += local_diff
            index_diff = abs(local_diff)
            self.assertLessEqual(index_diff, max_diff,
                                 msg=f"indexes shouldn't be off by more than {max_diff}. "
                                     f"probability_indexes: {prob_indexes} "
                                     f"count_indexes: {count_indexes}")
        self.assertLessEqual(
            abs(total_diff), max_diff,
            msg=f"Total bucket diff should be less than {max_diff}, but got {total_diff}."
        )
        self.assertEqual(count_indexes[-1], prob_indexes[-1],
                         "Both bucket counts should have the same final index.")

    def test_seattle_household_size_distro_honored(self):
        """
        This methods checks results from
        generate_household_sizes_from_fixed_pop_size for the seattle location.
        It checks against the house distribution obtained from
        get_seattle_household_size_distro and make sure that the most common
        household size should be the size with the highest probability.

        Returns:
            None
        """
        self.is_debugging = False
        hh_distro = self.get_seattle_household_size_distro()
        hh_sizes = sphh.generate_household_size_count_from_fixed_pop_size(500, hh_distro)

        hh_size_list = list(hh_sizes)  # Comes as np.ndarray
        fewest_houses = min(hh_size_list)
        fewest_index = hh_size_list.index(fewest_houses)

        most_houses = max(hh_size_list)
        most_index = hh_size_list.index(most_houses)

        highest_probability = max(hh_distro.values())
        lowest_probability = min(hh_distro.values())

        most_houses_probability = hh_distro[most_index + 1]  # hh_distro is 1 indexed
        fewest_houses_probability = hh_distro[fewest_index + 1]

        self.assertEqual(highest_probability, most_houses_probability,
                         msg="The most common household size should be the size with the highest probability")

        prob_bucket_list = list(hh_distro.values())
        self.verify_buckets(
            probability_buckets=prob_bucket_list,
            count_buckets=hh_size_list
        )

    def test_custom_household_size_distro_honored(self):
        """
        This methods checks results from
        generate_household_sizes_from_fixed_pop_size with customized
        distribution. It checks that the most common household size should be
        the size with the highest probability and also uses
        verify_portion_honored method for validation logic.

        Returns:
            None
        """
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
        hh_sizes = sp.generate_household_size_count_from_fixed_pop_size(500, custom_distro)

        hh_size_list = list(hh_sizes)  # Comes as np.ndarray
        fewest_houses = min(hh_size_list)
        fewest_index = hh_size_list.index(fewest_houses)

        most_houses = max(hh_size_list)
        most_index = hh_size_list.index(most_houses)

        highest_probability = max(custom_distro.values())
        lowest_probability = min(custom_distro.values())

        most_houses_probability = custom_distro[most_index + 1]  # hh_distro is 1 indexed
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
        """
        Test the result from method get_seattle_household_size.
        It checks that the key of househould size must be 1 to 7 and the total
        probability distribition adds to 1.

        Returns:
            None
        """
        hh_distro = self.get_seattle_household_size_distro()

        total_sizes = sum(hh_distro.values())
        self.assertNotIn(0, hh_distro,
                         msg="Each key is a household size, 0 shouldn't be here")
        for x in range(1, 8):
            self.assertIn(x, hh_distro,
                          msg=f"Households come in 1 to 7. Size {x} should be in here.")
        # rounding total_size to decimal places (default 7) must be equal to 1
        self.assertAlmostEqual(total_sizes, 1,
                               msg=f"This is a probability distribution that should add up to 1. Got "
                               f"{total_sizes} from this: {hh_distro}")
        if self.is_debugging:
            print(total_sizes)
            print(hh_distro)

    def get_seattle_gender_by_age(self):
        """
        Helper method for read_gender_fraction_by_age_bracket but currently
        deprecated.

        Returns:
            age brackets by genders
        """
        sea_sex_age_brackets = sp.read_gender_fraction_by_age_bracket(
            # datadir=sp.datadir,
            datadir = sp.settings.datadir,
            state_location=self.d_state_location,
            location=self.d_location,
            country_location=self.d_country_location
        )
        return sea_sex_age_brackets
        pass

    def get_seattle_age_brackets(self):
        """
        Helper method for read_age_bracket_distr.

        Returns:
            age distribution by brackets for location set as class variables
        """
        sea_age_brackets = spdd.read_age_bracket_distr(
            # sp.datadir,
            sp.settings.datadir,
            location=self.d_location,
            state_location=self.d_state_location,
            country_location=self.d_country_location
        )
        return sea_age_brackets
        pass

    def get_census_age_brackets(self):
        """
        Helper method for get_census_age_brackets.

        Returns:
            age brackets dictionary where keys are bracket index and values are
            list of ages.
        """
        census_age_brackets = sp.get_census_age_brackets(
            # sp.datadir,
            sp.settings.datadir,
            state_location=self.d_state_location,
            country_location=self.d_country_location
        )
        int_age_brackets = {}
        for k in census_age_brackets:
            int_age_brackets[k] = list(census_age_brackets[k])
        return int_age_brackets

    def bucket_population_counts(self, age_bracket_dict,
                                 ages_array):
        """
        Args:
            age_bracket_dict (dict): age bracket dictionary
            ages_array (np.ndarray): array of ages obtained from get_age_sex_n (now deprecated)

        Returns:
            list of age counts for each bucket.
        """
        age_bucket_counts = []
        for bucket in age_bracket_dict:
            # Create a list in age_bucket_counts that is empty
            tmp_bucket_count = 0
            # copy age_bracket_dict[thisn] to target_ages
            target_ages = age_bracket_dict[bucket]
            # Loop through every age in array
            for x in ages_array:
                # If ages_array[current] in target_ages
                if x in target_ages:
                    tmp_bucket_count += 1
            age_bucket_counts.append(tmp_bucket_count)
            if self.is_debugging:
                print(f"Bucket: {bucket} Target ages: {target_ages}")
                print(f"total found: {tmp_bucket_count}")
        return age_bucket_counts

    @unittest.skip("deprecated method get_age_sex_n")
    def test_seattle_age_sex_n(self):
        self.is_debugging = False
        sea_age_bracket_distro = self.get_seattle_age_brackets()
        sea_sex_age_brackets = self.get_seattle_gender_by_age()
        census_age_brackets = self.get_census_age_brackets()

        age_array, sex_array = sp.get_age_sex_n(
            gender_fraction_by_age=sea_sex_age_brackets,
            age_bracket_distr=sea_age_bracket_distro,
            age_brackets=census_age_brackets,
            n_people=500
        )
        if self.is_debugging:
            print(f"ages: {age_array}")
            print(f"age array length: {len(age_array)}")
        age_count_buckets = self.bucket_population_counts(
            age_bracket_dict=census_age_brackets,
            ages_array=age_array
        )
        self.verify_portion_honored(
            probability_buckets=sea_age_bracket_distro,
            count_buckets=age_count_buckets,
            portion=0.2
        )

    @unittest.skip("deprecated method get_age_sex_n")
    def test_get_age_sex_n_honors_ages(self):
        self.is_debugging = False
        age_probabilities = {
            0:  0.05,
            1:  0.10,
            2:  0.15,
            3:  0.20,
            4:  0.00,
            5:  0.05,
            6:  0.00,
            7:  0.09,
            8:  0.08,
            9:  0.07,
            10: 0.06,
            11: 0.05,
            12: 0.04,
            13: 0.03,
            14: 0.02,
            15: 0.01
        }
        sex_by_age_buckets = {}
        sex_by_age_buckets['male'] = {}
        sex_by_age_buckets['female'] = {}
        for x in range(0,16):
            sex_by_age_buckets['male'][x] = 0.5
            sex_by_age_buckets['female'][x] = 0.5
        age_brackets = self.get_census_age_brackets()
        age_array, sex_array = sp.get_age_sex_n(
            gender_fraction_by_age=sex_by_age_buckets,
            age_bracket_distr=age_probabilities,
            age_brackets=age_brackets,
            n_people=10000
        )
        age_count_buckets = self.bucket_population_counts(
            age_bracket_dict=age_brackets,
            ages_array=age_array
        )
        self.verify_portion_honored(
            probability_buckets=age_probabilities,
            count_buckets=age_count_buckets,
            portion=0.2
        )
        self.verify_portion_honored(
            probability_buckets=age_probabilities,
            count_buckets=age_count_buckets,
            portion=0.1
        )
        pass

    @unittest.skip("deprecated method get_age_sex_n")
    def test_get_age_sex_n_honors_sexes(self):
        self.is_debugging = False
        age_buckets = {}
        for x in range(0,10):
            age_buckets[x] = 0.0625  # 1/16 fun fact, works with floating point
        male_age_buckets = {
            0: 1.0,
            1: 0.7,
            2: 0.0,
            3: 0.4
        }
        for x in range(4, 7):
            male_age_buckets[x] = 1.0
        for x in range(7, 10):
            male_age_buckets[x] = 0.0
        female_age_buckets = {}
        for x in range(0, 10):
            female_age_buckets[x] = 1.0 - male_age_buckets[x]
        age_sex_buckets = {}
        age_sex_buckets['male'] = male_age_buckets
        age_sex_buckets['female'] = female_age_buckets
        age_brackets = {}
        for x in range(0, 10):
            age_brackets[x] = [i + (10 * x) for i in range(0,10)]
        age_array, sex_array = sp.get_age_sex_n(
            gender_fraction_by_age=age_sex_buckets,
            age_bracket_distr=age_buckets,
            age_brackets=age_brackets,
            n_people=10000
        )
        weighted_probability_buckets = {}
        total_weight = sum(male_age_buckets.values())
        for i in male_age_buckets:
            weighted_probability_buckets[i] = \
                male_age_buckets[i] / total_weight
        male_age_counts = {}
        for x in range(0,10):
            male_age_counts[x] = 0
        for x in range(0, len(sex_array)):
            if sex_array[x] == 1:
                bucket_index = (age_array[x] // 10)
                male_age_counts[bucket_index] += 1
        self.verify_portion_honored(
            probability_buckets=weighted_probability_buckets,
            count_buckets=list(male_age_counts.values()),
            portion=0.2
        )

    def test_generate_age_count(self):
        """
        Test generate_age_count method to Create age count from  randomly
        generated distribution and 5000 people validation logic is in
        verify_buckets which use chi-square test.

        Returns:
            None
        """
        # dist is the randomly generated distrubution with 20 brackets
        dist = np.random.random(20)
        dist /= dist.sum()
        generated = sp.generate_age_count(n=5000, age_distr=dist)
        self.verify_buckets(dist, list(generated.values()))

    @pytest.mark.skip  # separate method for households larger than 1 is deprecated and will be removed soon
    def test_generate_larger_household_sizes(self):
        """
        Test generate_larger_household_sizes method if hh_size =1, it expectes
        method to return an empty array, otherwise an array of counts which the
        total should match the the hh_size[1:].

        Returns:
            None
        """
        size1 = sp.generate_larger_household_sizes(hh_sizes=[1])
        self.assertEqual(len(size1), 0)
        for i in range(2, 10):
            size = np.random.randint(low=1, high=50, size=i)
            with self.subTest(size=size):
                print(f"hh_size:{size}")
                result = sp.generate_larger_household_sizes(hh_sizes=size)
                print(f"actual hh_size:{collections.Counter(size)}")
                self.assertEqual(sum(size[1:]), len(result))

    def test_generate_household_sizes(self):
        """
        Test generate_larger_household_sizes method if hh_size =1, it expectes
        method to return an empty array, otherwise an array of counts which the
        total should match the the hh_size[1:].

        Returns:
            None
        """
        size1 = sp.generate_household_sizes(hh_sizes=[])
        self.assertEqual(len(size1), 0)
        for i in range(2, 10):
            size = np.random.randint(low=1, high=50, size=i)
            with self.subTest(size=size):
                print(f"hh_size:{size}")
                result = sp.generate_household_sizes(hh_sizes=size)
                print(f"actual hh_size:{collections.Counter(size)}")
                self.assertEqual(sum(size), len(result))

    def test_generate_household_sizes_from_fixed_pop_size(self):
        """
        Test generate_household_sizes_from_fixed_pop_size the test data is
        specifically crafted to execute all conditional branches of the method.

        Returns:
            None
        """
        even_dist = {1: 0.2,
                     2: 0.2,
                     3: 0.2,
                     4: 0.2,
                     5: 0.2}
        # 900 is divisble by the expected value (3.0) but 901 is not
        # this creates test cases for N_gen = N and N_gen < N condition
        for i in [900, 901]:
            hh = sp.generate_household_size_count_from_fixed_pop_size(N=i, hh_size_distr=even_dist)
            # verify the total number of people matches N
            self.assertEqual(i, sum([(n+1)*hh[n] for n in range(0, len(hh))]))
            # verify distribution
            self.verify_buckets(even_dist.values(), hh)

        # slightly modify the distribution to create expected value = 2.91 which will round down to 2.9
        # and create N_gen > N condition
        uneven_dist = {1: 0.2,
                       2: 0.2,
                       3: 0.2,
                       4: 0.29,
                       5: 0.11}
        hh2 = sp.generate_household_size_count_from_fixed_pop_size(N=900, hh_size_distr=uneven_dist)
        self.assertEqual(900, sum([(n+1)*hh2[n] for n in range(0, len(hh2))]))
        self.verify_buckets(uneven_dist.values(), hh2)


if __name__ == "__main__":
    unittest.main()