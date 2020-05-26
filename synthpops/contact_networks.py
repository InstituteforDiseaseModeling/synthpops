"""
This module generates the household, school, and workplace contact networks.
"""

import os
from copy import deepcopy
from collections import Counter

import sciris as sc
import numpy as np
import pandas as pd

import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean

from . import base as spb
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contacts as spct


def generate_household_sizes(Nhomes, hh_size_distr):
    """
    Given a number of homes and a household size distribution, generate the number of homes of each size.

    Args:
        Nhomes (int)         : The number of homes.
        hh_size_distr (dict) : The distribution of household sizes.

    Returns:
        An array with the count of households of size s at index s-1.
    """
    max_size = max(hh_size_distr.keys())
    hh_sizes = np.random.multinomial(Nhomes, [hh_size_distr[s] for s in range(1, max_size+1)], size=1)[0]
    return hh_sizes


def generate_household_sizes_from_fixed_pop_size(N, hh_size_distr):
    """
    Given a number of people and a household size distribution, generate the number of homes of each size needed to place everyone in a household.

    Args:
        N      (int)         : The number of people in the population.
        hh_size_distr (dict) : The distribution of household sizes.

    Returns:
        An array with the count of households of size s at index s-1.
    """

    # Quickly produce number of expected households for a population of size N
    ss = np.sum([hh_size_distr[s] * s for s in hh_size_distr])
    f = N / np.round(ss, 1)
    hh_sizes = np.zeros(len(hh_size_distr))

    for s in hh_size_distr:
        hh_sizes[s-1] = int(hh_size_distr[s] * f)
    N_gen = np.sum([hh_sizes[s-1] * s for s in hh_size_distr], dtype=int)

    # Check what population size was created from the drawn count of household sizes
    people_to_add_or_remove = N_gen - N

    # did not create household sizes to match or exceed the population size so add count for households needed
    hh_size_keys = [k for k in hh_size_distr]
    hh_size_distr_array = [hh_size_distr[k] for k in hh_size_keys]
    if people_to_add_or_remove < 0:

        people_to_add = -people_to_add_or_remove
        while people_to_add > 0:
            new_household_size = np.random.choice(hh_size_keys, p=hh_size_distr_array)

            if new_household_size > people_to_add:
                new_household_size = people_to_add
            people_to_add -= new_household_size

            hh_sizes[new_household_size-1] += 1

    # created households that result in too many people
    elif people_to_add_or_remove > 0:
        people_to_remove = people_to_add_or_remove
        while people_to_remove > 0:

            new_household_size_to_remove = np.random.choice(hh_size_keys, p=hh_size_distr_array)
            if new_household_size_to_remove > people_to_remove:
                new_household_size_to_remove = people_to_remove

            people_to_remove -= new_household_size_to_remove
            hh_sizes[new_household_size_to_remove-1] -= 1

    hh_sizes = hh_sizes.astype(int)
    return hh_sizes


def get_totalpopsize_from_household_sizes(hh_sizes):
    """
    Sum the population of a specific household size from the count array.

    Args:
        hh_sizes (array): The count of household size s at index s-1.

    Returns:
        An integer indicating the total number of people in household size s.
    """
    return np.sum([hh_sizes[s] * (s+1) for s in range(len(hh_sizes))])


def generate_household_head_age_by_size(hha_by_size_counts, hha_brackets, hh_size, single_year_age_distr):
    """
    Generate the age of the head of the household, also known as the reference person of the household,
    conditional on the size of the household.

    Args:
        hha_by_size_counts (matrix)  : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)          : The age brackets for the heads of household.
        hh_size (int)                : The household size.
        single_year_age_distr (dict) : The age distribution.

    Returns:
        Age of the head of the household or reference person.
    """
    distr = hha_by_size_counts[hh_size-1, :]
    b = spsamp.sample_single_arr(distr)
    hha = spsamp.sample_from_range(single_year_age_distr, hha_brackets[b][0], hha_brackets[b][-1])

    return hha


def generate_living_alone(hh_sizes, hha_by_size_counts, hha_brackets, single_year_age_distr):
    """
    Generate the ages of those living alone.

    Args:
        hh_sizes (array)             : The count of household size s at index s-1.
        hha_by_size_counts (matrix)  : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)          : The age brackets for the heads of household.
        single_year_age_distr (dict) : The age distribution.

    Returns:
        An array of households of size 1 where each household is a row and the value in the row is the age of the household member.
    """

    size = 1
    homes = np.zeros((hh_sizes[size-1], 1), dtype=int)

    for h in range(hh_sizes[size-1]):
        hha = generate_household_head_age_by_size(hha_by_size_counts, hha_brackets, size, single_year_age_distr)
        homes[h][0] = hha

    return homes


def generate_larger_households(size, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr):
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
    ya_coin = 0.15  # produces far too few young adults without this for Seattle, Washington. This is a placeholder value. Users will need to change to fit whatever population they are working with.

    homes = np.zeros((hh_sizes[size-1], size), dtype=int)

    for h in range(hh_sizes[size-1]):

        hha = generate_household_head_age_by_size(hha_by_size_counts, hha_brackets, size, single_year_age_distr)

        homes[h][0] = hha

        b = age_by_brackets_dic[hha]
        b = min(b, contact_matrix_dic['H'].shape[0]-1) # Ensure it doesn't go past the end of the array
        b_prob = contact_matrix_dic['H'][b, :]

        age_distr_vals = np.array(list(single_year_age_distr.values()), dtype=np.float64) # Convert to an array for faster processing

        for n in range(1, size):
            bi = spsamp.sample_single_arr(b_prob)
            ai = spsamp.sample_from_range(single_year_age_distr, age_brackets[bi][0], age_brackets[bi][-1])

            if ai > 5 and ai <= 20:  # This a placeholder range. Users will need to change to fit whatever population they are working with.
                if np.random.binomial(1, ya_coin):
                    ai = spsamp.sample_from_range(single_year_age_distr, 25, 32)  # This a placeholder range. Users will need to change to fit whatever population they are working with.

            ai = spsamp.resample_age(age_distr_vals, ai)

            homes[h][n] = ai

    return homes


def generate_all_households(N, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr):
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
    homes_dic[1] = generate_living_alone(hh_sizes, hha_by_size_counts, hha_brackets, single_year_age_distr)
    # remove living alone from the distribution to choose from!
    for h in homes_dic[1]:
        single_year_age_distr[h[0]] -= 1.0/N

    for s in range(2, 8):
        homes_dic[s] = generate_larger_households(s, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr)

    homes = []
    for s in homes_dic:
        homes += list(homes_dic[s])

    np.random.shuffle(homes)
    return homes_dic, homes


def assign_uids_by_homes(homes, id_len=16, use_int=True):
    """
    Assign IDs to everyone in order by their households.

    Args:
        homes (array): The generated synthetic ages of household members.
        id_len (int) : The length of the UID.

    Returns:
        A copy of the generated households with IDs in place of ages, and a dictionary mapping ID to age.
    """
    age_by_uid_dic = {}
    homes_by_uids = []

    for h, home in enumerate(homes):

        home_ids = []
        for a in home:
            if use_int:
                uid = len(age_by_uid_dic)
            else:
                uid = sc.uuid(length=id_len)
            age_by_uid_dic[uid] = a
            home_ids.append(uid)

        homes_by_uids.append(home_ids)

    return homes_by_uids, age_by_uid_dic


def write_homes_by_age_and_uid(datadir, location, state_location, country_location, homes_by_uids, age_by_uid_dic):
    """
    Write the households to file with both ID and their ages, while also writing the dictionary of ID mapping to age for each individual in the population.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        homes_by_uids (list)      : The list of lists, where each sublist represents a household and the IDs of the household members.
        age_by_uid_dic (dict)     : A dictionary mapping ID to age for each individual in the population.

    Returns:
        None
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')
    os.makedirs(file_path, exist_ok=True)

    households_by_age_path = os.path.join(file_path, location + '_' + str(len(age_by_uid_dic)) + '_synthetic_households_with_ages.dat')
    households_by_uid_path = os.path.join(file_path, location + '_' + str(len(age_by_uid_dic)) + '_synthetic_households_with_uids.dat')
    age_by_uid_path = os.path.join(file_path, location + '_' + str(len(age_by_uid_dic)) + '_age_by_uid.dat')

    fh_age = open(households_by_age_path, 'w')
    fh_uid = open(households_by_uid_path, 'w')
    f_age_uid = open(age_by_uid_path, 'w')

    for n, ids in enumerate(homes_by_uids):

        home = homes_by_uids[n]

        for uid in home:

            fh_age.write(str(age_by_uid_dic[uid]) + ' ')
            fh_uid.write(str(uid) + ' ')
            f_age_uid.write(str(uid) + ' ' + str(age_by_uid_dic[uid]) + '\n')
        fh_age.write('\n')
        fh_uid.write('\n')
    fh_age.close()
    fh_uid.close()
    f_age_uid.close()


def read_in_age_by_uid(datadir, location, state_location, country_location, N):
    """
    Read dictionary of ID mapping to ages for all individuals from file.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        N (int)                   : The number of people in the population.
    Returns:
        A dictionary mapping ID to age for all individuals in the population.

    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')
    age_by_uid_path = os.path.join(file_path, location + '_' + str(N) + '_age_by_uid.dat')
    df = pd.read_csv(age_by_uid_path, header=None, delimiter=' ')
    return dict(zip(df.iloc[:, 0].values, df.iloc[:, 1].values))


def read_setting_groups(datadir, location, state_location, country_location, n, setting, with_ages=False):
    """
    Read in groups of people interacting in different social settings from file.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        n (int)                   : The number of people in the population.
        setting (string): The name of the physical contact setting: H for households, S for schools, W for workplaces, C for community or other.
        with_ages (bool): If True, read in the ages of each individual in the group; otherwise, read in their IDs.

    Returns:
        A list of lists where each sublist represents of group of individuals in the same group and thus are contacts of each other.
    """
    if with_ages:
        file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks', location + '_' + str(n) + '_synthetic_' + setting + '_with_ages.dat')
    else:
        file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks', location + '_' + str(n) + '_synthetic_' + setting + '_with_uids.dat')
    groups = []
    foo = open(file_path, 'r')
    for c, line in enumerate(foo):
        group = line.strip().split(' ')
        if with_ages:
            group = [int(a) for a in group]
        groups.append(group)
    return groups


def get_uids_in_school(datadir, n, location, state_location, country_location, age_by_uid_dic=None, homes_by_uids=None, use_default=False):
    """
    Identify who in the population is attending school based on enrollment rates by age.

    Args:
        datadir (string)          : The file path to the data directory.
        n (int)                   : The number of people in the population.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        age_by_uid_dic (dict)     : A dictionary mapping ID to age for all individuals in the population.
        homes_by_uids (list)      : A list of lists where each sublist is a household and the IDs of the household members.
        use_default (bool)        : If True, try to first use the other parameters to find data specific to the location under study; otherwise, return default data drawing from Seattle, Washington.

    Returns:
        A dictionary of students in schools mapping their ID to their age, a dictionary of students in school mapping age to the list of IDs with that age, and a dictionary mapping age to the number of students with that age.
    """
    uids_in_school = {}
    uids_in_school_by_age = {}
    ages_in_school_count = dict.fromkeys(np.arange(101), 0)

    rates = spdata.get_school_enrollment_rates(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    for a in np.arange(101):
        uids_in_school_by_age[a] = []

    if age_by_uid_dic is None:
        age_by_uid_dic = read_in_age_by_uid(datadir, location, state_location, country_location, n)

    if homes_by_uids is None:
        try:
            homes_by_uids = read_setting_groups(datadir, location, state_location, country_location, n, setting='households', with_ages=False)
        except:
            raise NotImplementedError('No households to bring in. Create people through those first.')

    # # go through all people at random and make a list of uids going to school as students
    # for uid in age_by_uid_dic:
    #     a = age_by_uid_dic[uid]
    #     if a <= 50:
    #         b = np.random.binomial(1,rates[a])
    #         if b:
    #             uids_in_school[uid] = a
    #             uids_in_school_by_age[a].append(uid)
    #             ages_in_school_count[a] += 1

    # go through homes and make a list of uids going to school as students, this should preserve ordering of students by homes and so create schools with siblings going to the same school
    for home in homes_by_uids:
        for uid in home:

            a = age_by_uid_dic[uid]
            if rates[a] > 0:
                b = np.random.binomial(1, rates[a])  # ask each person if they'll be a student - probably could be done in a faster, more aggregate way.
                if b:
                    uids_in_school[uid] = a
                    uids_in_school_by_age[a].append(uid)
                    ages_in_school_count[a] += 1

    return uids_in_school, uids_in_school_by_age, ages_in_school_count


def generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school):
    """
    Given a number of students in school, generate a list of school sizes to place everyone in a school.

    Args:
        school_size_distr_by_bracket (dict) : The distribution of binned school sizes.
        school_size_brackets (dict)         : A dictionary of school size brackets.
        uids_in_school (dict)               : A dictionary of students in school mapping ID to age.

    Returns:
        A list of school sizes whose sum is the length of ``uids_in_school``.
    """
    ns = len(uids_in_school)
    sorted_brackets = sorted(school_size_brackets.keys())
    prob_by_sorted_brackets = [school_size_distr_by_bracket[b] for b in sorted_brackets]

    school_sizes = []

    while ns > 0:
        size_bracket = np.random.choice(sorted_brackets, p=prob_by_sorted_brackets)
        # size = np.random.choice(school_size_brackets[size_bracket])  # creates some schools that are much smaller than expected so use average instead
        size = int(np.mean(school_size_brackets[size_bracket]))  # use average school size to avoid schools with very small sizes
        ns -= size
        school_sizes.append(size)
    if ns < 0:
        school_sizes[-1] = school_sizes[-1] + ns
    np.random.shuffle(school_sizes)
    return school_sizes


def send_students_to_school(school_sizes, uids_in_school, uids_in_school_by_age, ages_in_school_count, age_brackets, age_by_brackets_dic, contact_matrix_dic, verbose=False):
    """
    A method to send students to school together. Using the matrices to construct schools is not a perfect method so some things are more forced than the matrix method alone would create.

    Args:
        school_sizes (list): A list of school sizes.
        uids_in_school (dict): A dictionary of students in school mapping ID to age.
        uids_in_school_by_age (dict): A dictionary of students in school mapping age to the list of IDs with that age.
        ages_in_school_count (dict): A dictionary mapping age to the number of students with that age.
        age_brackets (dict)          : A dictionary mapping age bracket keys to age bracket range.
        age_by_brackets_dic (dict)   : A dictionary mapping age to the age bracket range it falls within.
        contact_matrix_dic (dict)    : A dictionary of age specific contact matrix for different physical contact settings.
        verbose (bool): If True, print statements about the generated schools as they're being generated.

    Returns:
        Two lists of lists, the first where each sublist is the ages of students in the same school, and the second is the same list but with the IDs of each student in place of their age.
    """
    syn_schools = []
    syn_school_uids = []

    ages_in_school_distr = spb.norm_dic(ages_in_school_count)
    left_in_bracket = spb.get_aggregate_ages(ages_in_school_count, age_by_brackets_dic)

    for n, size in enumerate(school_sizes):

        if len(uids_in_school) == 0:  # no more students left to send to school!
            break

        ages_in_school_distr = spb.norm_dic(ages_in_school_count)

        new_school = []
        new_school_uids = []

        achoice = np.random.multinomial(1, [ages_in_school_distr[a] for a in ages_in_school_distr])
        aindex = np.where(achoice)[0][0]
        bindex = age_by_brackets_dic[aindex]

        # reference students under 20 to prevent older adults from being reference students (otherwise we end up with schools with too many adults and kids mixing because the matrices represent the average of the patterns and not the bimodal mixing of adult students together at school and a small number of teachers at school with their students)
        if bindex >= 4:
            if np.random.binomial(1, p=0.7):
                achoice = np.random.multinomial(1, [ages_in_school_distr[a] for a in ages_in_school_distr])
                aindex = np.where(achoice)[0][0]

        uid = uids_in_school_by_age[aindex][0]
        uids_in_school_by_age[aindex].remove(uid)
        uids_in_school.pop(uid, None)
        ages_in_school_count[aindex] -= 1
        ages_in_school_distr = spb.norm_dic(ages_in_school_count)

        new_school.append(aindex)
        new_school_uids.append(uid)

        if verbose:
            print('reference school age', aindex, 'school size', size, 'students left', len(uids_in_school), left_in_bracket)

        bindex = age_by_brackets_dic[aindex]
        b_prob = contact_matrix_dic['S'][bindex, :]

        left_in_bracket[bindex] -= 1

        # fewer students than school size so everyone else is in one school
        if len(uids_in_school) < size:
            for uid in uids_in_school:
                ai = uids_in_school[uid]
                new_school.append(int(ai))
                new_school_uids.append(uid)
                uids_in_school_by_age[ai].remove(uid)
                ages_in_school_count[ai] -= 1
                left_in_bracket[age_by_brackets_dic[ai]] -= 1
            uids_in_school = {}
            if verbose:
                print('last school', 'size from distribution', size, 'size generated', len(new_school))

        else:
            bi_min = max(0, bindex-1)
            bi_max = bindex + 1

            for i in range(1, size):
                if len(uids_in_school) == 0:
                    break

                # no one left to send? should only choose other students from the mixing matrices, not teachers so don't create schools with
                if np.sum([left_in_bracket[bi] for bi in np.arange(bi_min, bi_max+1)]) == 0:
                    break

                bi = spsamp.sample_single_arr(b_prob)

                while left_in_bracket[bi] == 0 or np.abs(bindex - bi) > 1:
                    bi = spsamp.sample_single_arr(b_prob)

                ai = spsamp.sample_from_range(ages_in_school_distr, age_brackets[bi][0], age_brackets[bi][-1])
                uid = uids_in_school_by_age[ai][0]  # grab the next student in line

                new_school.append(ai)
                new_school_uids.append(uid)

                uids_in_school_by_age[ai].remove(uid)
                uids_in_school.pop(uid, None)

                ages_in_school_count[ai] -= 1
                ages_in_school_distr = spb.norm_dic(ages_in_school_count)
                left_in_bracket[bi] -= 1

        syn_schools.append(new_school)
        syn_school_uids.append(new_school_uids)
        new_school = np.array(new_school)
        kids = new_school <= 19
        # new_school_age_counter = Counter(new_school)
        if verbose:
            print('new school ages', len(new_school), sorted(new_school), 'nkids', kids.sum(), 'n20+', len(new_school)-kids.sum(), 'kid-adult ratio', kids.sum()/(len(new_school)-kids.sum()))
    if verbose:
        print('people in school', np.sum([len(school) for school in syn_schools]), 'left to send', len(uids_in_school))
    return syn_schools, syn_school_uids


def get_uids_potential_workers(syn_school_uids, employment_rates, age_by_uid_dic):
    """
    Get IDs for everyone who could be a worker by removing those who are students and those who can't be employed officially.

    Args:
        syn_school_uids (list)  : A list of lists where each sublist represents a school with the IDs of students in the school.
        employment_rates (dict) : The employment rates by age.
        age_by_uid_dic (dict)   : A dictionary mapping ID to age for individuals in the population.

    Returns:
        A dictionary of potential workers mapping their ID to their age, a dictionary mapping age to the list of IDs for potential
        workers with that age, and a dictionary mapping age to the count of potential workers left to assign to a workplace for that age.
    """
    potential_worker_uids = deepcopy(age_by_uid_dic)
    potential_worker_uids_by_age = {}
    potential_worker_ages_left_count = {}

    for a in range(101):
        if a >= 15:
            potential_worker_uids_by_age[a] = []
            potential_worker_ages_left_count[a] = 0

    for school in syn_school_uids:
        for uid in school:
            potential_worker_uids.pop(uid, None)

    for uid in age_by_uid_dic:
        if age_by_uid_dic[uid] not in employment_rates:
            potential_worker_uids.pop(uid, None)

    for uid in potential_worker_uids:
        ai = potential_worker_uids[uid]
        potential_worker_uids_by_age[ai].append(uid)
        potential_worker_ages_left_count[ai] += 1

    # shuffle workers around!
    for ai in potential_worker_uids_by_age:
        np.random.shuffle(potential_worker_uids_by_age[ai])

    return potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count


def generate_workplace_sizes(workplace_size_distr_by_bracket, workplace_size_brackets, workers_by_age_to_assign_count):
    """
    Given a number of individuals employed, generate a list of workplace sizes to place everyone in a workplace.

    Args:
        workplace_size_distr_by_bracket (dict) : The distribution of binned workplace sizes.
        worplace_size_brackets (dict)          : A dictionary of workplace size brackets.
        workers_by_age_to_assign_count (dict)  : A dictionary mapping age to the count of employed individuals of that age.

    Returns:
        A list of workplace sizes.
    """
    nworkers = np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count])

    # normalize workplace_size_distr_by_bracket because it's likely a count rather than distribution
    workplace_size_distr_by_bracket = spb.norm_dic(workplace_size_distr_by_bracket)

    sorted_brackets = sorted(workplace_size_brackets.keys())
    prob_by_sorted_brackets = [workplace_size_distr_by_bracket[b] for b in sorted_brackets]

    workplace_sizes = []

    while nworkers > 0:
        size_bracket = np.random.choice(sorted_brackets, p=prob_by_sorted_brackets)
        size = np.random.choice(workplace_size_brackets[size_bracket])
        nworkers -= size
        workplace_sizes.append(size)
    if nworkers < 0:
        workplace_sizes[-1] = workplace_sizes[-1] + nworkers
    np.random.shuffle(workplace_sizes)
    return workplace_sizes


def generate_usa_workplace_sizes(workplace_sizes_by_bracket, workplace_size_brackets, workers_by_age_to_assign_count):
    """
    Given a number of individuals employed, generate a list of workplace sizes to place everyone in a workplace.
    Specific to data from the US.

    Args:
        workplace_sizes_by_bracket (dict)     : The distribution of binned workplace sizes.
        worplace_size_brackets (dict)         : A dictionary of workplace size brackets.
        workers_by_age_to_assign_count (dict) : A dictionary mapping age to the count of employed individuals of that age.

    Returns:
        A list of workplace sizes.
    """
    nw = np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count])

    size_distr = {}
    for b in workplace_size_brackets:
        size = int(np.mean(workplace_size_brackets[b]) + 0.5)
        size_distr[size] = workplace_sizes_by_bracket[b]

    size_distr = spb.norm_dic(size_distr)
    workplace_sizes = []

    s_range = sorted(size_distr.keys())
    p = [size_distr[s] for s in s_range]

    while nw > 0:
        s = np.random.choice(s_range, p=p)
        nw -= s
        workplace_sizes.append(s)

    if nw < 0:
        workplace_sizes[-1] = workplace_sizes[-1] + nw

    np.random.shuffle(workplace_sizes)
    return workplace_sizes


def get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count, uids_by_age_dic):
    """
    Get the number of people to assign to a workplace by age using those left who can potentially go to work and employment rates by age.

    Args:
        employment_rates (dict)                 : A dictionary of employment rates by age.
        potential_worker_ages_left_count (dict) : A dictionary of the count of workers to assign by age.
        uids_by_age_dic (dict)                  : A dictionary mapping age to the list of ids with that age.

    Returns:
        A dictionary with a count of workers to assign to a workplace.
    """

    workers_by_age_to_assign_count = dict.fromkeys(np.arange(101), 0)
    for a in potential_worker_ages_left_count:
        if a in employment_rates:
            try:
                c = int(employment_rates[a] * len(uids_by_age_dic[a]))
            except:
                c = 0
            number_of_people_who_can_be_assigned = min(c, potential_worker_ages_left_count[a])
            workers_by_age_to_assign_count[a] = number_of_people_who_can_be_assigned

    return workers_by_age_to_assign_count


def assign_teachers_to_work(syn_schools, syn_school_uids, employment_rates, workers_by_age_to_assign_count, potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count, student_teacher_ratio=30, teacher_age_min=25, teacher_age_max=75, verbose=False):
    """
    Assign teachers to each school according to the average student-teacher ratio.

    Args:
        syn_schools (list): list of lists where each sublist is a school with the ages of the students within
        syn_school_uids (list): list of lists where each sublist is a school with the ids of the students within
        employment_rates (dict): employment rates by age
        workers_by_age_to_assign_count (dict): dictionary of the count of workers left to assign by age
        potential_worker_uids (dict): dictionary of potential workers mapping their id to their age
        potential_worker_uids_by_age (dict): dictionary mapping age to the list of worker ids with that age
        potential_worker_ages_left_count (dict): dictionary of the count of potential workers left that can be assigned by age
        student_teacher_ratio (int): average student teacher ratio
        teacher_age_min (int): minimum age for teachers - should be location specific
        teacher_age_max (int): maximum age for teachers - should be location specific
        verbose (bool): If True, print statements about the generated schools as teachers are being added to each school.

    Returns:
        List of lists of schools with the ages of individuals in each, lists of lists of schools with the ids of individuals in each,
        dictionary of potential workers mapping id to their age, dictionary mapping age to the list of potential workers of that age,
        dictionary with the count of workers left to assign for each age after teachers have been assigned.
    """
    # matrix method will already get some teachers into schools so student_teacher_ratio should be higher

    all_teachers = dict.fromkeys(np.arange(101), 0)

    for n in range(len(syn_schools)):
        school = syn_schools[n]
        school_uids = syn_school_uids[n]

        size = len(school)
        nteachers = int(size/float(student_teacher_ratio))
        nteachers = max(1, nteachers)
        if verbose:
            print('nteachers', nteachers, 'student-teacher ratio', size/nteachers)
        teachers = []
        teacher_uids = []

        for nt in range(nteachers):

            a = spsamp.sample_from_range(workers_by_age_to_assign_count, teacher_age_min, teacher_age_max)
            uid = potential_worker_uids_by_age[a][0]
            teachers.append(a)
            all_teachers[a] += 1

            potential_worker_uids_by_age[a].remove(uid)
            workers_by_age_to_assign_count[a] -= 1
            potential_worker_ages_left_count[a] -= 1
            potential_worker_uids.pop(uid, None)

            school.append(a)
            school_uids.append(uid)
            teacher_uids.append(uid)

        syn_schools[n] = school
        syn_school_uids[n] = school_uids
        if verbose:
            print('school with teachers', sorted(school))
            print('nkids', (np.array(school) <= 19).sum(), 'n20+', (np.array(school) > 19).sum())
            print('kid-adult ratio', (np.array(school) <= 19).sum() / (np.array(school) > 19).sum())

    return syn_schools, syn_school_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count


def assign_rest_of_workers(workplace_sizes, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count, age_by_uid_dic, age_brackets, age_by_brackets_dic, contact_matrix_dic, verbose=False):
    """
    Assign the rest of the workers to non-school workplaces.

    Args:
        workplace_sizes (list)                : list of workplace sizes
        potential_worker_uids (dict)          : dictionary of potential workers mapping their id to their age
        potential_worker_uids_by_age (dict)   : dictionary mapping age to the list of worker ids with that age
        workers_by_age_to_assign_count (dict) : dictionary of the count of workers left to assign by age
        age_by_uid_dic (dict)                 : dictionary mapping id to age for all individuals in the population
        age_brackets (dict)                   : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)            : dictionary mapping age to the age bracket range it falls in
        contact_matrix_dic (dict)             : dictionary of age specific contact matrix for different physical contact settings
        verbose (bool)                        : If True, print statements about the generated schools as teachers are being added to each school.

    Returns:
        List of lists where each sublist is a workplace with the ages of workers, list of lists where each sublist is a workplace with the ids of workers,
        dictionary of potential workers left mapping id to age, dictionary mapping age to a list of potential workers left of that age, dictionary
        mapping age to the count of workers left to assign.
    """
    syn_workplaces = []
    syn_workplace_uids = []
    worker_age_keys = workers_by_age_to_assign_count.keys()
    sorted_worker_age_keys = sorted(worker_age_keys)

    # off turn likelihood to meet those unemployed in the workplace because the matrices are not an exact match for the population under study
    for b in age_brackets:
        workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[b]]
        number_of_workers_left_in_bracket = np.sum(workers_left_in_bracket)
        if number_of_workers_left_in_bracket == 0:
            b = min(b, contact_matrix_dic['W'].shape[1]-1) # Ensure it doesn't go past the end of the array
            contact_matrix_dic['W'][:, b] = 0

    for n, size in enumerate(workplace_sizes):
        workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)
        if np.sum([workers_by_age_to_assign_distr[a] for a in workers_by_age_to_assign_distr]) == 0:
            break
        if np.sum([len(potential_worker_uids_by_age[a]) for a in potential_worker_uids_by_age]) == 0:
            break
        new_work, new_work_uids = [], []

        a_prob = [workers_by_age_to_assign_count[a] for a in sorted_worker_age_keys]
        a_prob = np.array(a_prob)
        a_prob = a_prob/np.sum(a_prob)

        achoice = np.random.choice(a=sorted_worker_age_keys, p=a_prob)
        aindex = achoice

        uid = potential_worker_uids_by_age[aindex][0]
        potential_worker_uids_by_age[aindex].remove(uid)
        potential_worker_uids.pop(uid, None)
        workers_by_age_to_assign_count[aindex] -= 1
        workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)
        new_work.append(aindex)
        new_work_uids.append(uid)

        bindex = age_by_brackets_dic[aindex]
        bindex = min(bindex, contact_matrix_dic['W'].shape[0]-1) # Ensure it doesn't go past the end of the array
        b_prob = contact_matrix_dic['W'][bindex, :]
        if np.sum(b_prob) > 0:
            b_prob = b_prob/np.sum(b_prob)

        if size > len(potential_worker_uids)-1:
            size = len(potential_worker_uids)-1
        workers_left_count = np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count])
        if size > workers_left_count:
            size = workers_left_count+1

        # not enough people left over to try to match age mixing patterns in the last workplace so grab everyone who will get placed in order
        if len(potential_worker_uids) <= size or workers_left_count <= size:
            for ai in workers_by_age_to_assign_count:
                for i in range(workers_by_age_to_assign_count[ai]):  # do not change this during the loop but afterwards, and if 0 then no one will be placed
                    uid = potential_worker_uids_by_age[ai][0]
                    new_work.append(ai)
                    new_work_uids.append(uid)
                    potential_worker_uids_by_age[ai].remove(uid)
                    potential_worker_uids.pop(uid, None)
                workers_by_age_to_assign_count[ai] = 0  # set to zero now that everyone will be placed in this last workplace
            workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)
        else:
            for i in range(1, size):

                bichoice = np.random.multinomial(1, b_prob)
                bi = np.where(bichoice)[0][0]

                workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[bi] if len(potential_worker_uids_by_age[a]) > 0]
                if np.sum(b_prob):
                    while np.sum(workers_left_in_bracket) == 0:
                        bichoice = np.random.multinomial(1, b_prob)
                        bi = np.where(bichoice)[0][0]
                        workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[bi] if len(potential_worker_uids_by_age[a]) > 0]
                    a_prob = [workers_by_age_to_assign_count[a] for a in age_brackets[bi]]
                    a_prob = np.array(a_prob)
                    a_prob = a_prob/np.sum(a_prob)

                    ai = np.random.choice(a=age_brackets[bi], p=a_prob)

                    uid = potential_worker_uids_by_age[ai][0]
                    new_work.append(ai)
                    new_work_uids.append(uid)
                    potential_worker_uids_by_age[ai].remove(uid)
                    potential_worker_uids.pop(uid, None)
                    workers_by_age_to_assign_count[ai] -= 1
                    workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)

                # if there's no one left in the bracket, then you should turn this bracket off in the contact matrix
                workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[bi]]
                if np.sum(workers_left_in_bracket) == 0:
                    contact_matrix_dic['W'][:, bi] = 0.
                    # since the matrix was modified, calculate the bracket probabilities again
                    b_prob = contact_matrix_dic['W'][bindex, :]
                    if np.sum(b_prob) > 0:
                        b_prob = b_prob/np.sum(b_prob)

        if verbose:
            print(n, Counter(new_work))

        syn_workplaces.append(new_work)
        syn_workplace_uids.append(new_work_uids)
    return syn_workplaces, syn_workplace_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count


def write_schools_by_age_and_uid(datadir, location, state_location, country_location, n, schools_by_uids, age_by_uid_dic):
    """
    Write the schools to file with both id and their ages.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        schools_by_uids (list)    : list of lists, where each sublist represents a school and the ids of the students and teachers within it
        age_by_uid_dic (dict)     : dictionary mapping id to age for each individual in the population

    Returns:
        None
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')
    os.makedirs(file_path, exist_ok=True)
    schools_by_age_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_schools_with_ages.dat')
    schools_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_schools_with_uids.dat')

    fh_age = open(schools_by_age_path, 'w')
    fh_uid = open(schools_by_uid_path, 'w')

    for n, ids in enumerate(schools_by_uids):

        school = schools_by_uids[n]
        for uid in school:

            fh_age.write(str(age_by_uid_dic[uid]) + ' ')
            fh_uid.write(str(uid) + ' ')
        fh_age.write('\n')
        fh_uid.write('\n')
    fh_age.close()
    fh_uid.close()


def write_workplaces_by_age_and_uid(datadir, location, state_location, country_location, n, workplaces_by_uids, age_by_uid_dic):
    """
    Write the workplaces to file with both id and their ages.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        workplaces_by_uids (list) : list of lists, where each sublist represents a workplace and the ids of the workers within it
        age_by_uid_dic (dict)     : dictionary mapping id to age for each individual in the population

    Returns:
        None
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')
    os.makedirs(file_path, exist_ok=True)
    workplaces_by_age_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_with_ages.dat')
    workplaces_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_with_uids.dat')

    fh_age = open(workplaces_by_age_path, 'w')
    fh_uid = open(workplaces_by_uid_path, 'w')

    for n, ids in enumerate(workplaces_by_uids):

        work = workplaces_by_uids[n]

        for uid in work:

            fh_age.write(str(age_by_uid_dic[uid]) + ' ')
            fh_uid.write(str(uid) + ' ')
        fh_age.write('\n')
        fh_uid.write('\n')
    fh_age.close()
    fh_uid.close()


def generate_synthetic_population(n, datadir, location='seattle_metro', state_location='Washington', country_location='usa', sheet_name='United States of America', school_enrollment_counts_available=False, verbose=False, plot=False, write=False, return_popdict=False, use_default=False):
    """
    Wrapper function that calls other functions to generate a full population with their contacts in the household, school, and workplace layers,
    and then writes this population to appropriate files.

    Args:
        n (int)                                   : The number of people in the population.
        datadir (string)                          : The file path to the data directory.
        location (string)                         : The name of the location.
        state_location (string)                   : The name of the state the location is in.
        country_location (string)                 : The name of the country the location is in.
        sheet_name (string)                       : The name of the sheet in the Excel file with contact patterns.
        school_enrollment_counts_available (bool) : If True, a list of school sizes is available and a count of the sizes can be constructed.
        verbose (bool)                            : If True, print statements as contacts are being generated.
        plot (bool)                               : If True, plot and show a comparison of the generated age distribution in households vs. the expected age distribution of the population from census data being sampled.
        write (bool)                              : If True, write population to file.
        return_popdict (bool)                     : If True, returns a dictionary of individuals in the population
        use_default (bool)                        : If True, try to first use the other parameters to find data specific to the location under study; otherwise, return default data drawing from Seattle, Washington.

    Returns:
        If return_popdict is True, returns popdict, a dictionary of people with attributes. Dictionary keys are the IDs of individuals in the population and the values are a dictionary for each individual with their
        attributes, such as age, household ID (hhid), school ID (scid), workplace ID (wpid), workplace industry code (wpindcode) if available, and the IDs of their contacts in different layers. Different layers
        available are households ('H'), schools ('S'), and workplaces ('W'). Contacts in these layers are clustered and thus form a network composed of groups of people interacting with each other. For example, all
        household members are contacts of each other, and everyone in the same school is a contact of each other. Else, return None.

    Example
    =======

    ::

        datadir = sp.datadir # point datadir where your data folder lives

        location = 'seattle_metro'
        state_location = 'Washington'
        country_location = 'usa'
        sheet_name = 'United States of America'

        n = 10000
        verbose = False
        plot = False

        # this will generate a population with microstructure and age demographics that
        # approximate those of the location selected
        # also saves to file in:
        #    datadir/demographics/contact_matrices_152_countries/state_location/
        sp.generate_synthetic_population(n,datadir,location=location,
                                         state_location=state_location,
                                         country_location=country_location,
                                         sheet_name=sheet_name,verbose=verbose,plot=plot)
    """
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location, use_default=use_default)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    contact_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name=sheet_name, use_default=use_default)

    household_size_distr = spdata.get_household_size_distr(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    min_pop = 100
    if n < min_pop:
        raise NotImplementedError(f"Population is too small to currently be generated properly. Try a size larger than {min_pop}.")
    n = int(n)

    # this could be unnecessary if we get the single year age distribution in a different way.
    n_to_sample_smoothly = int(1e6)
    hh_sizes = generate_household_sizes(n_to_sample_smoothly, household_size_distr)
    totalpop = get_totalpopsize_from_household_sizes(hh_sizes)

    # create a rough single year age distribution to draw from instead of the distribution by age brackets.
    syn_ages, syn_sexes = spsamp.get_usa_age_sex_n(datadir, location, state_location, country_location, totalpop)
    syn_age_count = Counter(syn_ages)
    syn_age_distr_unordered = spb.norm_dic(syn_age_count) # Ensure it's ordered
    syn_age_keys = list(syn_age_distr_unordered.keys())
    sort_inds = np.argsort(syn_age_keys)
    syn_age_distr = {}
    for i in sort_inds:
        syn_age_distr[syn_age_keys[i]] = syn_age_distr_unordered[syn_age_keys[i]]

    # actual household sizes
    hh_sizes = generate_household_sizes_from_fixed_pop_size(n, household_size_distr)
    totalpop = get_totalpopsize_from_household_sizes(hh_sizes)

    hha_brackets = spdata.get_head_age_brackets(datadir, country_location=country_location, use_default=use_default)
    hha_by_size = spdata.get_head_age_by_size_distr(datadir, country_location=country_location, use_default=use_default)

    homes_dic, homes = generate_all_households(n, hh_sizes, hha_by_size, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, deepcopy(syn_age_distr))
    homes_by_uids, age_by_uid_dic = assign_uids_by_homes(homes)
    new_ages_count = Counter(age_by_uid_dic.values())

    # plot synthetic age distribution as a check
    if plot:

        cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
        cmap3 = mplt.cm.get_cmap(cmocean.cm.matter)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        x = np.arange(101)
        y_exp = np.zeros(101)
        y_sim = np.zeros(101)

        for a in range(101):
            expected = int(syn_age_distr[a] * totalpop)
            y_exp[a] = expected
            y_sim[a] = new_ages_count[a]

        ax.plot(x, y_exp, color=cmap(0.2), label='Expected')
        ax.plot(x, y_sim, color=cmap3(0.6), label='Simulated')
        leg = ax.legend(fontsize=18)
        leg.draw_frame(False)
        ax.set_xlim(left=0, right=100)
        ax.set_xticks(np.arange(0, 101, 5))

        plt.show()

    # Make a dictionary listing out uids of people by their age
    uids_by_age_dic = spb.get_ids_by_age_dic(age_by_uid_dic)

    # Generate school sizes
    school_sizes_count_by_brackets = spdata.get_school_size_distr_by_brackets(datadir, location=location, state_location=state_location, country_location=country_location, counts_available=school_enrollment_counts_available, use_default=use_default)
    school_size_brackets = spdata.get_school_size_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    # Figure out who's going to school as a student with enrollment rates (gets called inside sp.get_uids_in_school)
    uids_in_school, uids_in_school_by_age, ages_in_school_count = get_uids_in_school(datadir, n, location, state_location, country_location, age_by_uid_dic, homes_by_uids, use_default=use_default)  # this will call in school enrollment rates

    # Get school sizes
    gen_school_sizes = generate_school_sizes(school_sizes_count_by_brackets, school_size_brackets, uids_in_school)

    # Assign students to school
    gen_schools, gen_school_uids = send_students_to_school(gen_school_sizes, uids_in_school, uids_in_school_by_age, ages_in_school_count, age_brackets, age_by_brackets_dic, contact_matrix_dic, verbose)

    # Get employment rates
    employment_rates = spdata.get_employment_rates(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    # Find people who can be workers (removing everyone who is currently a student)
    potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count = get_uids_potential_workers(gen_school_uids, employment_rates, age_by_uid_dic)
    workers_by_age_to_assign_count = get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count, uids_by_age_dic)

    # Assign teachers and update school lists
    gen_schools, gen_school_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count = assign_teachers_to_work(gen_schools, gen_school_uids, employment_rates, workers_by_age_to_assign_count, potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count, verbose=verbose)

    # Generate non-school workplace sizes needed to send everyone to work
    workplace_size_brackets = spdata.get_workplace_size_brackets(datadir, state_location=state_location, country_location=country_location, use_default=use_default)
    workplace_size_distr_by_brackets = spdata.get_workplace_size_distr_by_brackets(datadir, state_location=state_location, country_location=country_location, use_default=use_default)
    workplace_sizes = generate_workplace_sizes(workplace_size_distr_by_brackets, workplace_size_brackets, workers_by_age_to_assign_count)

    verbose = False
    if verbose:
        for a in employment_rates:
            print(a, workers_by_age_to_assign_count[a]/len(uids_by_age_dic[a]), employment_rates[a])

    # Assign all workers who are not staff at schools to workplaces
    gen_workplaces, gen_workplace_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count = assign_rest_of_workers(workplace_sizes, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count, age_by_uid_dic, age_brackets, age_by_brackets_dic, contact_matrix_dic, verbose=verbose)

    workers_placed_by_age_count = dict.fromkeys(np.arange(0, 101), 0)
    for w in gen_workplaces:
        for a in w:
            workers_placed_by_age_count[a] += 1

    if verbose:
        for a in workers_placed_by_age_count:
            print(a, workers_placed_by_age_count[a], int(employment_rates[a] * len(uids_by_age_dic[a])), workers_placed_by_age_count[a]/len(uids_by_age_dic[a]), employment_rates[a], workers_placed_by_age_count[a]/len(uids_by_age_dic[a])/employment_rates[a])
        print('workers left to place', np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count]))
        print('work sizes made', np.sum([len(w) for w in gen_workplaces]))

    # save schools and workplace uids to file
    if write:
        write_homes_by_age_and_uid(datadir, location, state_location, country_location, homes_by_uids, age_by_uid_dic)
        write_schools_by_age_and_uid(datadir, location, state_location, country_location, n, gen_school_uids, age_by_uid_dic)
        write_workplaces_by_age_and_uid(datadir, location, state_location, country_location, n, gen_workplace_uids, age_by_uid_dic)

    if return_popdict:
        popdict = spct.make_contacts_from_microstructure_objects(age_by_uid_dic, homes_by_uids, gen_school_uids, gen_workplace_uids)
        return popdict
