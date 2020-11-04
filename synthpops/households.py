'''
Functions for generating households
'''

import sciris as sc
import numpy as np
from . import sampling as spsamp


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
        homes[h][0] = int(hha)

    return homes


def assign_uids_by_homes(homes, id_len=16, use_int=True):
    """
    Assign IDs to everyone in order by their households.

    Args:
        homes (array)  : The generated synthetic ages of household members.
        id_len (int)   : The length of the UID.
        use_int (bool) : If True, use ints for the uids of individuals; otherwise use strings of length 'id_len'.

    Returns:
        A copy of the generated households with IDs in place of ages, and a dictionary mapping ID to age.
    """
    age_by_uid_dic = dict()
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