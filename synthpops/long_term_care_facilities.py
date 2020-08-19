"""
Modeling Seattle Metro Long Term Care Facilities

"""

import numpy as np

import os
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter

from . import base as spb
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contacts as spct
from . import contact_networks as spcnx
from . import school_modules as spsm
from . import read_write as sprw

part = 2

# Customized age resampling method
def custom_resample_age(exp_age_distr, a):
    """
    Resampling younger ages to better match data

    Args:
        single_year_age_distr (dict) : age distribution
        age (int)                    : age as an integer
    Returns:
        Resampled age as an integer.

    Notes:
        This is not always necessary, but is mostly used to smooth out sharp edges in the age distribution when spsamp.resample_age() produces too many of one year and under produces the surrounding ages. For example, new borns (0 years old) may
        be over produced, and 1 year olds under produced, so this function can be customized to correct for that.
    """
    # exp_age_distr = np.array(list(exp_age_distr_dict.values()), dtype=np.float64)
    a = spsamp.resample_age(exp_age_distr, a)
    if a == 7:
        if np.random.binomial(1, p=0.25):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 6:
        if np.random.binomial(1, p=0.25):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 5:
        if np.random.binomial(1, p=0.2):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 0:
        if np.random.binomial(1, p=0.0):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 1:
        if np.random.binomial(1, p=0.1):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 2:
        if np.random.binomial(1, p=0.0):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 4:
        if np.random.binomial(1, p=0.1):
            a = spsamp.resample_age(exp_age_distr, a)
    return a


# Customized household construction methods
def custom_generate_larger_households(size, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr):
    """
    Generate ages of those living in households of greater than one individual. Reference individual is sampled conditional on the household size.
    All other household members have their ages sampled conditional on the reference person's age and the age mixing contact matrix
    in households for the population under study.

    Args:
        size (int)                   : The household size.
        hh_sizes (array)             : The count of household size s at index s-1.
        hha_by_size_counts (matrix)  : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)          : The age brackets for the heads of household.
        age_brackets (dict)          : A dictionary mapping age bracket keys to age bracket range.
        age_by_brackets_dic (dict)   : A dictionary mapping age to the age bracket range it falls within.
        contact_matrix_dic (dict)    : A dictionary of the age-specific contact matrix for different physical contact settings.
        single_year_age_distr (dict) : The age distribution.

    Returns:
        An array of households for size ``size`` where each household is a row and the values in the row are the ages of the household members.
        The first age in the row is the age of the reference individual.
    """
    ya_coin = 0.15  # This is a placeholder value. Users will need to change to fit whatever population you are working with

    homes = np.zeros((hh_sizes[size-1], size), dtype=int)

    for h in range(hh_sizes[size-1]):

        hha = spcnx.generate_household_head_age_by_size(hha_by_size_counts, hha_brackets, size, single_year_age_distr)

        homes[h][0] = hha

        b = age_by_brackets_dic[hha]
        b = min(b, contact_matrix_dic['H'].shape[0]-1) # Ensure it doesn't go past the end of the array
        b_prob = contact_matrix_dic['H'][b, :]

        for n in range(1, size):
            bi = spsamp.sample_single_arr(b_prob)
            ai = spsamp.sample_from_range(single_year_age_distr, age_brackets[bi][0], age_brackets[bi][-1])

            """ The following is an example of how you may resample from an age range that is over produced and instead
                sample ages from an age range that is under produced in your population. This kind of customization may
                be necessary when your age mixing matrix and the population you are interested in modeling differ in
                important but subtle ways. For example, generally household age mixing matrices reflect mixing patterns
                for households composed of families. This means household age mixing matrices do not generally cover
                college or university aged individuals living together. Without this customization, this algorithm tends
                to under produce young adults. This method also has a tendency to underproduce the elderly, and does not
                explicitly model the elderly living in nursing homes. Customizations like this should be considered in
                context of the specific population and culture you are trying to model. In some cultures, it is common to
                live in non-family households, while in others family households are the most common and include
                multi-generational family households. If you are unsure of how to proceed with customizations please
                take a look at the references listed in the overview documentation for more information.
            """
            if ai > 5 and ai <= 20:  # This a placeholder range. Users will need to change to fit whatever population you are working with
                if np.random.binomial(1, ya_coin):
                    ai = spsamp.sample_from_range(single_year_age_distr, 25, 32)  # This is a placeholder range. Users will need to change to fit whatever populaton you are working with

            # ai = spsamp.resample_age(single_year_age_distr, ai)
            ai = custom_resample_age(single_year_age_distr, ai)

            homes[h][n] = ai

    return homes


def custom_generate_all_households(N, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr):
    """
    Generate the ages of those living in households together. First create households of people living alone, then larger households.
    For households larger than 1, a reference individual's age is sampled conditional on the household size, while all other household
    members have their ages sampled conditional on the reference person's age and the age mixing contact matrix in households
    for the population under study.

    Args:
        N (int)                      : The number of people in the population.
        hh_sizes (array)             : The count of household size s at index s-1.
        hha_by_size_counts (matrix)  : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)          : The age brackets for the heads of household.
        age_brackets (dict)          : The dictionary mapping age bracket keys to age bracket range.
        age_by_brackets_dic (dict)   : The dictionary mapping age to the age bracket range it falls within.
        contact_matrix_dic (dict)    : The dictionary of the age-specific contact matrix for different physical contact settings.
        single_year_age_distr (dict) : The age distribution.

    Returns:
        An array of all households where each household is a row and the values in the row are the ages of the household members.
        The first age in the row is the age of the reference individual. Households are randomly shuffled by size.
    """

    homes_dic = {}
    homes_dic[1] = spcnx.generate_living_alone(hh_sizes, hha_by_size_counts, hha_brackets, single_year_age_distr)
    # remove living alone from the distribution to choose from!
    for h in homes_dic[1]:
        single_year_age_distr[h[0]] -= 1.0/N

    # generate larger households and the ages of people living in them
    for s in range(2, len(hh_sizes) + 1):
        homes_dic[s] = custom_generate_larger_households(s, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr)

    homes = []
    for s in homes_dic:
        homes += list(homes_dic[s])

    np.random.shuffle(homes)
    return homes_dic, homes


def generate_microstructure_with_facilities(datadir, location, state_location, country_location, n, sheet_name='United States of America',
                                            use_two_group_reduction=False, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
                                            school_enrollment_counts_available=False, with_school_types=False, school_mixing_type='random',average_class_size=20, inter_grade_mixing=0.1,
                                            average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
                                            average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
                                            verbose=False, plot=False, write=False, return_popdict=False, use_default=False):

    # Grab Long Term Care Facilities data
    ltcf_df = spdata.get_usa_long_term_care_facility_data(datadir, state_location, part)

    # ltcf_df keys
    ltcf_age_bracket_keys = ['Under 65', '65–74', '75–84', '85 and over']
    facility_keys = [
                    # 'Hospice',
                    'Nursing home',
                    'Residential care community'
                    ]

    # state numbers
    facillity_users = {}
    for fk in facility_keys:
        facillity_users[fk] = {}
        facillity_users[fk]['Total'] = int(ltcf_df[ltcf_df.iloc[:, 0] == 'Number of users2, 5'][fk].values[0].replace(',', ''))
        for ab in ltcf_age_bracket_keys:
            facillity_users[fk][ab] = float(ltcf_df[ltcf_df.iloc[:, 0] == ab][fk].values[0].replace(',', ''))/100.

    total_facility_users = np.sum([facillity_users[fk]['Total'] for fk in facillity_users])

    # Census Bureau numbers 2016
    state_pop_2016 = 7288000
    state_age_distr_2016 = {}
    state_age_distr_2016['60-64'] = 6.3
    state_age_distr_2016['65-74'] = 9.0
    state_age_distr_2016['75-84'] = 4.0
    state_age_distr_2016['85-100'] = 1.8

    # Census Bureau numbers 2018
    state_pop_2018 = 7535591
    state_age_distr_2018 = {}
    state_age_distr_2018['60-64'] = 6.3
    state_age_distr_2018['65-74'] = 9.5
    state_age_distr_2018['75-84'] = 4.3
    state_age_distr_2018['85-100'] = 1.8

    for a in state_age_distr_2016:
        state_age_distr_2016[a] = state_age_distr_2016[a]/100.
        state_age_distr_2018[a] = state_age_distr_2018[a]/100.

    num_state_elderly_2016 = 0
    num_state_elderly_2018 = 0
    for a in state_age_distr_2016:
        num_state_elderly_2016 += state_pop_2016 * state_age_distr_2016[a]
        num_state_elderly_2018 += state_pop_2018 * state_age_distr_2018[a]

    expected_users_2018 = total_facility_users * num_state_elderly_2018/num_state_elderly_2016

    if verbose:
        print('number of elderly',num_state_elderly_2016, num_state_elderly_2018)
        print('growth in elderly', num_state_elderly_2018/num_state_elderly_2016)
        print('users in 2016',total_facility_users, '% of elderly', total_facility_users/num_state_elderly_2016)
        print('users in 2018', expected_users_2018)

    # location age distribution
    age_distr_16 = spdata.read_age_bracket_distr(datadir, country_location=country_location, state_location=state_location, location=location)
    age_brackets_16 = spdata.get_census_age_brackets(datadir, state_location, country_location)
    age_by_brackets_dic_16 = spb.get_age_by_brackets_dic(age_brackets_16)

    # current King County population size
    pop = 2.25e6

    # local elderly population estimate
    local_elderly_2018 = 0
    for ab in range(12, 16):
        local_elderly_2018 += age_distr_16[ab] * pop

    if verbose:
        print('number of local elderly', local_elderly_2018)

    # growth_since_2016 = num_state_elderly_2018/num_state_elderly_2016
    # local_perc_elderly_2018 = local_elderly_2018/num_state_elderly_2018

    if verbose:
        print('local users in 2018?', total_facility_users * local_elderly_2018/num_state_elderly_2018 * num_state_elderly_2018/num_state_elderly_2016)
    # seattle_users_est_from_state = total_facility_users * local_perc_elderly_2018 * growth_since_2016

    est_seattle_users_2018 = dict.fromkeys(['60-64', '65-74', '75-84', '85-100'], 0)

    for fk in facillity_users:
        for ab in facillity_users[fk]:
            if ab != 'Total':
                # print(fk, ab, facillity_users[fk][ab], facillity_users[fk][ab] * facillity_users[fk]['Total'], facillity_users[fk][ab] * facillity_users[fk]['Total'] * pop/state_pop_2018)
                if ab == 'Under 65':
                    b = '60-64'
                elif ab == '65–74':
                    b = '65-74'
                elif ab == '75–84':
                    b = '75-84'
                elif ab == '85 and over':
                    b = '85-100'
                est_seattle_users_2018[b] += facillity_users[fk][ab] * facillity_users[fk]['Total'] * pop/state_pop_2018

    if verbose:
        for ab in est_seattle_users_2018:
            print(ab, est_seattle_users_2018[ab], est_seattle_users_2018[ab]/(state_age_distr_2018[ab] * pop))
        print(np.sum([est_seattle_users_2018[b] for b in est_seattle_users_2018]))

    # for pop of 2.25 million of Seattle
    est_ltcf_user_by_age_brackets_perc = {}
    for b in est_seattle_users_2018:
        est_ltcf_user_by_age_brackets_perc[b] = est_seattle_users_2018[b]/state_age_distr_2018[b]/pop
        # print(b,est_ltcf_user_by_age_brackets_perc[b])

    est_ltcf_user_by_age_brackets_perc['65-69'] = est_ltcf_user_by_age_brackets_perc['65-74']
    est_ltcf_user_by_age_brackets_perc['70-74'] = est_ltcf_user_by_age_brackets_perc['65-74']
    est_ltcf_user_by_age_brackets_perc['75-79'] = est_ltcf_user_by_age_brackets_perc['75-84']
    est_ltcf_user_by_age_brackets_perc['80-84'] = est_ltcf_user_by_age_brackets_perc['75-84']

    est_ltcf_user_by_age_brackets_perc.pop('65-74', None)
    est_ltcf_user_by_age_brackets_perc.pop('75-84', None)

    age_distr_18_fp = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions', 'seattle_metro_age_bracket_distr_18.dat')
    age_distr_18 = spdata.read_age_bracket_distr(datadir, file_path=age_distr_18_fp)
    age_brackets_18_fp = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions', 'census_age_brackets_18.dat')
    age_brackets_18 = spdata.get_census_age_brackets(datadir, file_path=age_brackets_18_fp)
    age_by_brackets_dic_18 = spb.get_age_by_brackets_dic(age_brackets_18)

    n = int(n)

    expected_users_by_age = {}

    for a in range(60, 101):
        if a < 65:
            b = age_by_brackets_dic_18[a]

            expected_users_by_age[a] = n * age_distr_18[b] / len(age_brackets_18[b])
            expected_users_by_age[a] = expected_users_by_age[a] * est_ltcf_user_by_age_brackets_perc['60-64']
            expected_users_by_age[a] = int(math.ceil(expected_users_by_age[a]))

        elif a < 75:
            b = age_by_brackets_dic_18[a]

            expected_users_by_age[a] = n * age_distr_18[b] / len(age_brackets_18[b])
            expected_users_by_age[a] = expected_users_by_age[a] * est_ltcf_user_by_age_brackets_perc['70-74']
            expected_users_by_age[a] = int(math.ceil(expected_users_by_age[a]))

        elif a < 85:
            b = age_by_brackets_dic_18[a]

            expected_users_by_age[a] = n * age_distr_18[b] / len(age_brackets_18[b])
            expected_users_by_age[a] = expected_users_by_age[a] * est_ltcf_user_by_age_brackets_perc['80-84']
            expected_users_by_age[a] = int(math.ceil(expected_users_by_age[a]))

        elif a < 101:
            b = age_by_brackets_dic_18[a]

            expected_users_by_age[a] = n * age_distr_18[b] / len(age_brackets_18[b])
            expected_users_by_age[a] = expected_users_by_age[a] * est_ltcf_user_by_age_brackets_perc['85-100']
            expected_users_by_age[a] = int(math.ceil(expected_users_by_age[a]))

    if verbose:
        print(np.sum([expected_users_by_age[a] for a in expected_users_by_age]))

    KC_resident_size_distr = spdata.get_usa_long_term_care_facility_residents_distr(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)
    KC_resident_size_distr = spb.norm_dic(KC_resident_size_distr)
    KC_residents_size_brackets = spdata.get_usa_long_term_care_facility_residents_distr_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    all_residents = []
    for a in expected_users_by_age:
        all_residents += [a] * expected_users_by_age[a]
    np.random.shuffle(all_residents)

    # place residents in facilities
    facilities = []

    size_bracket_keys = sorted([k for k in KC_resident_size_distr.keys()])
    size_distr_array = [KC_resident_size_distr[k] for k in size_bracket_keys]
    while len(all_residents) > 0:

        sb = np.random.choice(size_bracket_keys, p=size_distr_array)
        sb_range = KC_residents_size_brackets[sb]
        size = np.random.choice(sb_range)

        # size = int(np.random.choice(KC_ltcf_sizes))
        if size > len(all_residents):
            size = len(all_residents)

        new_facility = all_residents[0:size]
        facilities.append(new_facility)
        all_residents = all_residents[size:]

    max_age = 100

    expected_age_distr = dict.fromkeys(np.arange(max_age + 1), 0)
    expected_age_count = dict.fromkeys(np.arange(max_age + 1), 0)

    # adjust age distribution for those already created
    for a in expected_age_distr:
        expected_age_distr[a] = age_distr_16[age_by_brackets_dic_16[a]]/len(age_brackets_16[age_by_brackets_dic_16[a]])
        expected_age_count[a] = int(n * expected_age_distr[a])

    ltcf_adjusted_age_count = deepcopy(expected_age_count)
    for a in expected_users_by_age:
        ltcf_adjusted_age_count[a] -= expected_users_by_age[a]
    ltcf_adjusted_age_distr_dict = spb.norm_dic(ltcf_adjusted_age_count)
    ltcf_adjusted_age_distr = np.array([ltcf_adjusted_age_distr_dict[i] for i in range(max_age+1)])

    exp_age_distr = np.array([expected_age_distr[i] for i in range(max_age+1)], dtype=np.float64)
    # exp_age_distr = np.array(list(expected_age_distr.values()), dtype=np.float64)

    # build rest of the population
    n_nonltcf = n - np.sum([len(f) for f in facilities])  # remove those placed in care homes

    household_size_distr = spdata.get_household_size_distr(datadir, location, state_location, country_location, use_default=use_default)
    hh_sizes = spcnx.generate_household_sizes_from_fixed_pop_size(n_nonltcf, household_size_distr)
    hha_brackets = spdata.get_head_age_brackets(datadir, country_location=country_location, use_default=use_default)
    hha_by_size = spdata.get_head_age_by_size_distr(datadir, country_location=country_location, use_default=use_default)

    contact_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name=sheet_name)

    homes_dic, homes = custom_generate_all_households(n_nonltcf, hh_sizes, hha_by_size, hha_brackets, age_brackets_16, age_by_brackets_dic_16, contact_matrix_dic, ltcf_adjusted_age_distr)
    homes = facilities + homes

    homes_by_uids, age_by_uid_dic = spcnx.assign_uids_by_homes(homes)  # include facilities to assign ids
    new_ages_count = Counter(age_by_uid_dic.values())

    facilities_by_uids = homes_by_uids[0:len(facilities)]

    if plot:

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        x = np.arange(max_age+1)
        y_exp = np.zeros(max_age+1)
        y_sim = np.zeros(max_age+1)
        for a in x:
            y_exp[a] = expected_age_distr[a]
            y_sim[a] = new_ages_count[a]/n
        ax.plot(x, y_exp, color='k', label='Expected')
        ax.plot(x, y_sim, color='teal', label='Simulated')
        leg = ax.legend(fontsize=18)
        leg.draw_frame(False)
        ax.set_xlim(0, max_age+1)
        for a in range(6):
            ax.axvline(x=a, ymin=0, ymax=1)
        plt.show()

    # Make a dictionary listing out uids of people by their age
    uids_by_age_dic = spb.get_ids_by_age_dic(age_by_uid_dic)

    # Generate school sizes
    school_sizes_count_by_brackets = spdata.get_school_size_distr_by_brackets(datadir, location=location, state_location=state_location, country_location=country_location, counts_available=school_enrollment_counts_available, use_default=use_default)
    school_size_brackets = spdata.get_school_size_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    # Figure out who's going to school as a student with enrollment rates (gets called inside sp.get_uids_in_school)
    uids_in_school, uids_in_school_by_age, ages_in_school_count = spcnx.get_uids_in_school(datadir, n_nonltcf, location, state_location, country_location, age_by_uid_dic, homes_by_uids, use_default=use_default)  # this will call in school enrollment rates

    if with_school_types:

        school_size_distr_by_type = spsm.get_default_school_size_distr_by_type()
        school_size_brackets = spsm.get_default_school_size_distr_brackets()

        school_types_by_age = spsm.get_default_school_types_by_age()
        school_type_age_ranges = spsm.get_default_school_type_age_ranges()

        syn_schools, syn_school_uids, syn_school_types = spcnx.send_students_to_school_with_school_types(school_size_distr_by_type, school_size_brackets, uids_in_school, uids_in_school_by_age,
                                                                                                         ages_in_school_count,
                                                                                                         school_types_by_age,
                                                                                                         school_type_age_ranges,
                                                                                                         verbose=verbose)

    else:
        # use contact matrices to send students to school

        # Get school sizes
        syn_school_sizes = spcnx.generate_school_sizes(school_sizes_count_by_brackets, school_size_brackets, uids_in_school)

        # Assign students to school
        syn_schools, syn_school_uids, syn_school_types = spcnx.send_students_to_school(syn_school_sizes, uids_in_school, uids_in_school_by_age, ages_in_school_count, age_brackets_16, age_by_brackets_dic_16, contact_matrix_dic, verbose)

    # Get employment rates
    employment_rates = spdata.get_employment_rates(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    # Find people who can be workers (removing everyone who is currently a student)
    potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count = spcnx.get_uids_potential_workers(syn_school_uids, employment_rates, age_by_uid_dic)
    workers_by_age_to_assign_count = spcnx.get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count, uids_by_age_dic)

    # Removing facilities residents from potential workers
    for nf, fc in enumerate(facilities_by_uids):
        for uid in fc:
            aindex = age_by_uid_dic[uid]
            if uid in potential_worker_uids:
                potential_worker_uids_by_age[aindex].remove(uid)
                potential_worker_uids.pop(uid, None)
                if workers_by_age_to_assign_count[aindex] > 0:
                    workers_by_age_to_assign_count[aindex] -= 1

    # Assign teachers and update school lists
    syn_teachers, syn_teacher_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count = spcnx.assign_teachers_to_schools(syn_schools, syn_school_uids, employment_rates, workers_by_age_to_assign_count, potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count,
                                                                                                                                                           average_student_teacher_ratio=average_student_teacher_ratio, teacher_age_min=teacher_age_min, teacher_age_max=teacher_age_max, verbose=verbose)

    syn_non_teaching_staff_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count = spcnx.assign_additional_staff_to_schools(syn_school_uids, syn_teacher_uids, workers_by_age_to_assign_count, potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count,
                                                                                                                                                                average_student_teacher_ratio=average_student_teacher_ratio, average_student_all_staff_ratio=average_student_all_staff_ratio, staff_age_min=staff_age_min, staff_age_max=staff_age_max, verbose=verbose)

    # Assign facilities care staff from 20 to 59

    KC_ratio_distr = spdata.get_usa_long_term_care_facility_resident_to_staff_ratios_distr(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)
    KC_ratio_distr = spb.norm_dic(KC_ratio_distr)
    KC_ratio_brackets = spdata.get_usa_long_term_care_facility_resident_to_staff_ratios_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    facilities_staff = []
    facilities_staff_uids = []

    sorted_ratio_keys = sorted([k for k in KC_ratio_distr.keys()])
    sorted_ratio_array = [KC_ratio_distr[k] for k in sorted_ratio_keys]

    staff_age_range = np.arange(ltcf_staff_age_min, ltcf_staff_age_max + 1)
    for nf, fc in enumerate(facilities):
        n_residents = len(fc)

        sb = np.random.choice(sorted_ratio_keys, p=sorted_ratio_array)
        sb_range = KC_ratio_brackets[sb]
        resident_staff_ratio = np.mean(sb_range)

        # if using raw staff totals in residents to staff ratios divide rato by 3 to split staff into 3 8 hour shifts at minimum
        resident_staff_ratio = resident_staff_ratio/3.
        # resident_staff_ratio = np.random.choice(KC_resident_staff_ratios)

        n_staff = int(math.ceil(n_residents/resident_staff_ratio))
        new_staff, new_staff_uids = [], []

        for i in range(n_staff):
            a_prob = np.array([workers_by_age_to_assign_count[a] for a in staff_age_range])
            a_prob = a_prob/np.sum(a_prob)
            aindex = np.random.choice(a=staff_age_range, p=a_prob)

            uid = potential_worker_uids_by_age[aindex][0]
            potential_worker_uids_by_age[aindex].remove(uid)
            potential_worker_uids.pop(uid, None)
            workers_by_age_to_assign_count[aindex] -= 1

            new_staff.append(aindex)
            new_staff_uids.append(uid)

        facilities_staff.append(new_staff)
        facilities_staff_uids.append(new_staff_uids)

    if verbose:
        print(len(facilities_staff_uids))
        for nf, fc in enumerate(facilities):
            print(fc, facilities_staff[nf], len(fc)/len(facilities_staff[nf]))

    # Generate non-school workplace sizes needed to send everyone to work
    workplace_size_brackets = spdata.get_workplace_size_brackets(datadir, state_location=state_location, country_location=country_location, use_default=use_default)
    workplace_size_distr_by_brackets = spdata.get_workplace_size_distr_by_brackets(datadir, state_location=state_location, country_location=country_location, use_default=use_default)
    workplace_sizes = spcnx.generate_workplace_sizes(workplace_size_distr_by_brackets, workplace_size_brackets, workers_by_age_to_assign_count)

    # Assign all workers who are not staff at schools to workplaces
    syn_workplaces, syn_workplace_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count = spcnx.assign_rest_of_workers(workplace_sizes, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count, age_by_uid_dic, age_brackets_16, age_by_brackets_dic_16, contact_matrix_dic, verbose=verbose)

    # remove facilities from homes to write households as a separate file
    homes_by_uids = homes_by_uids[len(facilities_by_uids):]
    # group uids to file
    folder_name = 'contact_networks_facilities'
    if write:
        sprw.write_age_by_uid_dic(datadir, location, state_location, country_location, folder_name, age_by_uid_dic)

        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'households', homes_by_uids)
        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'schools', syn_school_uids)
        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'teachers', syn_teacher_uids)
        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'non_teaching_staff', syn_non_teaching_staff_uids)
        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'workplaces', syn_workplace_uids)
        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'facilities', facilities_by_uids)
        sprw.write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, 'facilities_staff', facilities_staff_uids)

    print('facilities_staff_uids', facilities_staff_uids)
    popdict = spct.make_contacts_with_facilities_from_microstructure_objects(age_by_uid_dic,
                                                                             homes_by_uids,
                                                                             syn_school_uids,
                                                                             syn_teacher_uids,
                                                                             syn_workplace_uids,
                                                                             facilities_by_uids,
                                                                             facilities_staff_uids,
                                                                             syn_non_teaching_staff_uids,
                                                                             use_two_group_reduction=use_two_group_reduction,
                                                                             average_LTCF_degree=average_LTCF_degree,
                                                                             with_school_types=with_school_types,
                                                                             school_mixing_type=school_mixing_type,
                                                                             average_class_size=average_class_size,
                                                                             inter_grade_mixing=inter_grade_mixing,
                                                                             average_student_teacher_ratio=average_student_teacher_ratio,
                                                                             average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                                             average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                                             average_additional_staff_degree=average_additional_staff_degree)

    if verbose:
        uids = popdict.keys()
        uids = [uid for uid in uids]
        np.random.shuffle(uids)

        for i in range(50):
            uid = uids[i]
            person = popdict[uid]
            print('uid', uid, person['age'], person['contacts']['H'], person['contacts']['S'], person['contacts']['W'], person['contacts']['LTCF'])
            print(person['snf_res'], person['snf_staff'])

    if return_popdict:
        return popdict


def check_all_residents_are_connected_to_staff(popdict):
    flag = True
    for i in popdict:
        person = popdict[i]
        if person['snf_res'] == 1:

            contacts = person['contacts']['LTCF']
            staff_contacts = [j for j in contacts if popdict[j]['snf_staff'] == 1]

            if len(staff_contacts) == 0:
                flag = False
                print('i', person['snf_res'], [popdict[j]['snf_staff'] for j in person['contacts']['LTCF']])
                errormsg = f'At least one LTCF or Skilled Nursing Facility resident has no contacts with staff members.'
                raise ValueError(errormsg)

    if flag:
        print('All LTCF residents have at least one contact with a staff member.')
