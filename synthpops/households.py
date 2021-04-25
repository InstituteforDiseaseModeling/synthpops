'''
Functions for generating households
'''

import sciris as sc
import numpy as np
from collections import Counter
from .config import logger as log, checkmem
from . import base as spb
from . import sampling as spsamp
from . import ltcfs as spltcf


def generate_household_size_count_from_fixed_pop_size(N, hh_size_distr):
    """
    Given a number of people and a household size distribution, generate the number of homes of each size needed to place everyone in a household.

    Args:
        N      (int)         : The number of people in the population.
        hh_size_distr (dict) : The distribution of household sizes.

    Returns:
        An array with the count of households of size s at index s-1.
    """
    log.debug('generate_household_size_count_from_fixed_pop_size()')
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
    log.debug('assign_uids_by_homes()')
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


def generate_age_count(n, age_distr):
    """
    Generate a stochastic count of people for each age given the age
    distribution (age_distr) and number of people to generate (n).

    Args:
        n (int)                        : number of people to generate
        age_distr (list or np.ndarray) : single year age distribution

    Returns:
        dict: A dictionary with the count of people to generate for each age
        given an age distribution and the number of people to generate.
    """
    log.debug('generate_age_count()')
    age_range = np.arange(0, len(age_distr))
    chosen = np.random.choice(age_range, size=n, p=age_distr)
    age_count = Counter(chosen)
    age_count = sc.mergedicts(dict.fromkeys(age_range, 0), age_count)
    return age_count


def generate_age_count_multinomial(n, age_distr):
    """
    Generate a stochastic count of people for each age given the age
    distribution (age_distr) and number of people to generate (n).

    Args:
        n (int)                        : number of people to generate
        age_distr (list or np.ndarray) : single year age distribution

    Returns:
        dict: A dictionary with the count of people to generate for each age
        given an age distribution and the number of people to generate.
    """
    log.debug('generate_age_count_multinomial()')
    age_count = np.random.multinomial(n, age_distr)
    return dict(zip(range(len(age_distr)), age_count))


def generate_household_head_ages(household_sizes, hha_by_size, hha_brackets, ages_left_to_assign):
    """
    Generate the head of household ages conditional on household size and the
    expected ages of people in the population.

    Args:
        household_sizes (np.array) : Array of household sizes to be generated
        hha_by_size (matrix)       : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)        : The age brackets for the heads of household.
        ages_left_to_assign (dic)  : The counter of ages for the generated population left to place in a residence

    Returns:
        An array of head of household ages, updated counter of the ages in the
        population left to place in a residence.
    """
    household_head_ages = []

    for nh, hs in enumerate(household_sizes):
        hs_distr = hha_by_size[hs - 1, :]
        hbi = spsamp.fast_choice(hs_distr)
        hbi_distr = np.array([ages_left_to_assign[a] for a in hha_brackets[hbi]])

        while sum(hbi_distr) == 0: # pragma: no cover
            hbi = spsamp.fast_choice(hs_distr)
            hbi_distr = np.array([ages_left_to_assign[a] for a in hha_brackets[hbi]])

        hha = hha_brackets[hbi][spsamp.fast_choice(hbi_distr)]
        ages_left_to_assign[hha] -= 1

        household_head_ages.append(hha)
    household_head_ages = np.array(household_head_ages).astype(int)

    return household_head_ages, ages_left_to_assign


def generate_household_sizes(hh_sizes):
    """
    Create a list of the household sizes in random order so that as individuals
    are placed by age into homes running out of specific ages is not
    systemically an issue for any given household size unless certain sizes
    greatly outnumber households of other sizes.

    Args:
        hh_sizes (array) : The count of household size s at index s-1.

    Returns:
        Np.array: An array of household sizes to be generated and place people
        into households.
    """
    household_sizes = []
    for hs in range(1, len(hh_sizes) + 1):
        household_sizes.extend([hs] * hh_sizes[hs - 1])
    household_sizes = np.array(household_sizes)
    np.random.shuffle(household_sizes)
    return household_sizes


def generate_larger_households_method_2(larger_hh_size_array, larger_hha_chosen, hha_brackets, cm_age_brackets, cm_age_by_brackets_dic, household_matrix, ages_left_to_assign, homes_dic):
    """
    Assign people to households larger than one person (excluding special
    residences like long term care facilities or agricultural workers living in
    shared residential quarters).

    Args:
        hh_sizes (array)              : The count of household size s at index s-1.
        hha_by_size (matrix)          : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)           : The age brackets for the heads of household.
        cm_age_brackets (dict)        : The age brackets for the contact matrix.
        cm_age_by_brackets_dic (dict) : A dictionary mapping age to the age bracket range it falls within.
        household_matrix (dict)       : The age-specific contact matrix for the household ontact setting.
        ages_left_to_assign (dict)    : Age count of people left to place in households larger than one person.

    Returns:
        dict: A dictionary of households by age indexed by household size.
    """

    # go through every household and assign the ages of the other household members from those left to place
    for nh, hs in enumerate(larger_hh_size_array):

        hha = larger_hha_chosen[nh]
        b = cm_age_by_brackets_dic[hha]

        home = np.zeros(hs)
        home[0] = hha

        for nj in range(1, hs):

            # can no longer place anyone in households where b is the age bracket of the head since those people are no longer available
            if np.sum(household_matrix[b, :]) == 0: # pragma: no cover
                break

            bi = spsamp.fast_choice(household_matrix[b, :])

            a_prob = np.array([ages_left_to_assign[a] for a in cm_age_brackets[bi]])
            if np.sum(a_prob) == 0:  # must check if all zeros since sp.fast_choice will not check
                household_matrix[:, bi] = 0  # turn off this part of the matrix

            # entire matrix has been turned off, can no longer select anyone
            if np.sum(household_matrix) == 0: # pragma: no cover
                break

            # must check if all zeros since sp.fast_choice will not check
            while np.sum(a_prob) == 0: # pragma: no cover
                bi = spsamp.fast_choice(household_matrix[b, :])
                a_prob = np.array([ages_left_to_assign[a] for a in cm_age_brackets[bi]])

                # must check if all zeros sine sp.fast_choice will not check
                if np.sum(a_prob) == 0: # pragma: no cover
                    household_matrix[:, bi] = 0

            aj = cm_age_brackets[bi][spsamp.fast_choice(a_prob)]
            ages_left_to_assign[aj] -= 1

            home[nj] = aj
        homes_dic[hs].append(home)

    for hs in homes_dic:
        homes_dic[hs] = np.array(homes_dic[hs]).astype(int)

    # at this point everyone should have been placed into a home
    sum_remaining = sum(ages_left_to_assign.values())
    assert sum_remaining== 0, f'Check failed: generating larger households method 2. {sum_remaining} and {ages_left_to_assign}.'
    return homes_dic, ages_left_to_assign


def generate_all_households_method_2(n_remaining, hh_sizes, hha_by_size, hha_brackets, cm_age_brackets, cm_age_by_brackets_dic, contact_matrices, ages_left_to_assign):
    """
    Generate the ages of those living in households together. First create
    households of people living alone, then larger households. For households
    larger than 1, a reference individual's age is sampled conditional on the
    household size, while all other household members have their ages sampled
    conditional on the reference person's age and the age mixing contact matrix
    in households for the population under study. Fix the count of ages in the
    population before placing individuals in households so that the age
    distribution of the generated population is fixed to closely match the age
    distribution from data on the population.

    Args:
        n_remaining (int)             : The number of people in the population left to place in a residence.
        hh_sizes (array)              : The count of household size s at index s-1.
        hha_by_size_counts (matrix)   : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)           : The age brackets for the heads of household.
        cm_age_brackets (dict)        : The dictionary mapping age bracket keys to age bracket range matching the household contact matrix.
        cm_age_by_brackets_dic (dict) : The dictionary mapping age to the age bracket range it falls within matching the household contact matrix.
        contact_matrix_dic (dict)     : The dictionary of the age-specific contact matrix for different physical contact settings.
        ages_left_to_assign (dict)    : Age count of people left to place in households larger than one person.

    Returns:
        An array of all households where each household is a row and the values
        in the row are the ages of the household members. The first age in the
        row is the age of the reference individual. Households are randomly
        shuffled by size.
    """
    log.debug('generate_all_households_method_2()')
    household_sizes = generate_household_sizes(hh_sizes)

    # generate the ages for heads of households or reference persons conditional on the household size and the age distribution
    household_head_ages, ages_left_to_assign = generate_household_head_ages(household_sizes, hha_by_size, hha_brackets, ages_left_to_assign)

    homes_dic = dict()

    # find all of the people living alone and their ages
    living_alone = list(household_head_ages[household_sizes == 1])
    homes_dic[1] = np.array(living_alone).astype(int).reshape((len(living_alone), 1))  # make an array of each individual home

    # arrays of the larger household sizes and the ages of the heads or reference persons for each (reference person ages generated conditional on household size and age distribution)
    larger_household_sizes = household_sizes[household_sizes > 1]
    heads_of_larger_households = household_head_ages[household_sizes > 1]

    for size in range(2, len(hh_sizes) + 1):
        homes_dic[size] = []

    # work off a copy of the household mixing matrix
    household_matrix = sc.dcp(contact_matrices['H'])

    homes_dic, ages_left_to_assign = generate_larger_households_method_2(larger_household_sizes, heads_of_larger_households, hha_brackets, cm_age_brackets, cm_age_by_brackets_dic, household_matrix, ages_left_to_assign, homes_dic)
    homes = get_all_households(homes_dic)

    return homes_dic, homes


def generate_larger_households_method_1(size, larger_household_sizes, heads_of_larger_households, hha_brackets, cm_age_brackets, cm_age_by_brackets_dic, household_matrix, adjusted_age_dist, p=0.15):
    """
    Generate ages of those living in households of greater than one individual.
    Reference individual is sampled conditional on the household size. All other
    household members have their ages sampled conditional on the reference
    person's age and the age mixing contact matrix in households for the
    population under study.

    Args:
        size (int)                    : The household size.
        hh_sizes (array)              : The count of household size s at index s-1.
        hha_by_size_counts (matrix)   : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)           : The age brackets for the heads of household.
        cm_age_brackets (dict)        : The dictionary mapping age bracket keys to age bracket range matching the household contact matrix.
        cm_age_by_brackets_dic (dict) : The dictionary mapping age to the age bracket range it falls within matching the household contact matrix.
        contact_matrix_dic (dict)     : A dictionary of the age-specific contact matrix for different physical contact settings.
        single_year_age_distr (dict)  : The age distribution.

    Returns:
        An array of households for size ``size`` where each household is a row
        and the values in the row are the ages of the household members. The
        first age in the row is the age of the reference individual.
    """
    log.debug('generate_larger_households_method_1()')
    # calibrated to work for Seattle metro - use method 2 for other populations
    # p = 0.15  # This is a placeholder value. Users will need to change this to fit whatever population they are working with if using this method

    household_size_mask = larger_household_sizes == size
    households_of_this_size = larger_household_sizes[household_size_mask]
    heads_of_this_size = heads_of_larger_households[household_size_mask]

    homes = np.zeros((len(households_of_this_size), size), dtype=int)

    for nh in range(len(households_of_this_size)):

        hha = heads_of_this_size[nh]

        homes[nh][0] = hha

        b = cm_age_by_brackets_dic[hha]
        b = min(b, household_matrix.shape[0] - 1)  # Ensure it doesn't go past the end of the array - likely not needed
        b_prob = household_matrix[b, :]

        for n in range(1, size):
            bi = spsamp.sample_single_arr(b_prob)
            ai = spsamp.sample_from_range(adjusted_age_dist, cm_age_brackets[bi][0], cm_age_brackets[bi][-1])  # sample from a range, defining the probabilities of the distribution and the minimum and maximum of the range

            if ai > 5 and ai <= 20:  # This is a placeholder range. Users will need to change this to fit their whatever population they are working with if using this method
                if np.random.binomial(1, p):
                    ai = spsamp.sample_from_range(adjusted_age_dist, 25, 32)

            ai = spltcf.ltcf_resample_age(adjusted_age_dist, ai)

            homes[nh][n] = ai

    return homes


def generate_all_households_method_1(n, n_remaining, hh_sizes, hha_by_size, hha_brackets, cm_age_brackets, cm_age_by_brackets_dic, contact_matrices, adjusted_age_dist, ages_left_to_assign):
    """
    Generate the ages of those living in households together. First create
    households of people living alone, then larger households. For households
    larger than 1, a reference individual's age is sampled conditional on the
    household size, while all other household members have their ages sampled
    conditional on the reference person's age and the age mixing contact matrix
    in households for the population under study.

    Args:
        n (int)                       : The number of people in the population.
        n_remaining (int)             : The number of people in the population left to place in a residence.
        hh_sizes (array)              : The count of household size s at index s-1.
        hha_by_size_counts (matrix)   : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)           : The age brackets for the heads of household.
        cm_age_brackets (dict)        : The dictionary mapping age bracket keys to age bracket range matching the household contact matrix.
        cm_age_by_brackets_dic (dict) : The dictionary mapping age to the age bracket range it falls within matching the household contact matrix.
        contact_matrices (dict)       : The dictionary of the age-specific contact matrix for different physical contact settings.
        ages_left_to_assign (dict)    : Age count of people left to place in households larger than one person.

    Returns:
        An array of all households where each household is a row and the values
        in the row are the ages of the household members. The first age in the
        row is the age of the reference individual. Households are randomly
        shuffled by size.

    Note:
        This method is not guaranteed to model the population age distribution
        well automatically. The method called inside,
        generate_larger_households_method_1 uses the method ltcf_resample_age to
        fit Seattle, Washington populations with long term care facilities
        generated. For a method that matches the age distribution well for
        populations in general, please use generate_all_households_methods_2.

        The following contains an example of how you may resample from an age
        range that is over produced and instead sample ages from an age range
        that is under produced in your population. This kind of customization
        may be necessary when your age mixing matrix and the population you are
        interested in modeling differ in important but subtle ways. For example,
        generally household age mixing matrices reflect mixing patterns for
        households composed of families. This means household age mixing
        matrices do not generally cover college or university aged individuals
        living together. Without this customization, this algorithm tends to
        under produce young adults. This method also has a tendency to
        underproduce the elderly, and does not explicitly model the elderly
        living in nursing homes. Customizations like this should be considered
        in context of the specific population and culture you are trying to
        model. In some cultures, it is common to live in non-family households,
        while in others family households are the most common and include
        multi-generational family households. If you are unsure of how to
        proceed with customizations please take a look at the references listed
        in the overview documentation for more information.
    """
    log.debug('generate_all_households_method_1()')
    household_sizes = generate_household_sizes(hh_sizes)

    # generate the ages for heads of households or reference persons conditional on the household size and the age distribution
    household_head_ages, ages_left_to_assign = generate_household_head_ages(household_sizes, hha_by_size, hha_brackets, ages_left_to_assign)

    homes_dic = dict()

    # find all of the people living alone and their ages
    living_alone = list(household_head_ages[household_sizes == 1])
    homes_dic[1] = np.array(living_alone).astype(int).reshape((len(living_alone), 1))  # make an array of each individual home

    # arrays of the larger household sizes and the ages of the heads or reference persons for each (reference person ages generated conditional on household size and age distribution)
    larger_household_sizes = household_sizes[household_sizes > 1]
    heads_of_larger_households = household_head_ages[household_sizes > 1]

    for size in range(2, len(hh_sizes) + 1):
        homes_dic[size] = []

    # work off a copy of the household mixing matrix
    household_matrix = sc.dcp(contact_matrices['H'])

    # remove the ages of household heads or reference persons already places
    for hha in household_head_ages:
        adjusted_age_dist[hha] -= 1 / n
    for a in adjusted_age_dist:
        adjusted_age_dist[a] = max(adjusted_age_dist[a], 0)
    adjusted_age_dist_values = np.array([adjusted_age_dist[a] for a in adjusted_age_dist])

    # generate the large households and ages of those people
    for size in range(2, len(hh_sizes) + 1):
        homes_dic[size] = generate_larger_households_method_1(size, larger_household_sizes, heads_of_larger_households, hha_brackets, cm_age_brackets, cm_age_by_brackets_dic, household_matrix, adjusted_age_dist_values)

    homes = get_all_households(homes_dic)

    return homes_dic, homes


def get_all_households(homes_dic):
    """
    Get all households in a list, randomly assorted.

    Args:
        homes_dic (dict): A dictionary of households by age indexed by household size

    Returns:
        list: A random ordering of households with the ages of the individuals.
    """
    homes = []
    for hs in homes_dic:
        homes.extend(homes_dic[hs])

    np.random.shuffle(homes)
    return homes


def get_household_sizes(popdict):
    """
    Get household sizes for each household in the popdict.

    Args:
        popdict (dict) : population dictionary

    Returns:
        dict: Dictionary of the generated household size for each household.
    """
    household_sizes = dict()
    for i, person in popdict.items():
        if person['hhid'] is not None:
            household_sizes.setdefault(person['hhid'], 0)
            household_sizes[person['hhid']] += 1

    return household_sizes


def get_household_heads(popdict):
    """
    Get the id of the head of each household.

    Args:
        popdict (dict) : population dictionary

    Returns:
        dict: Dictionary of the id of the head of the household for each household.
    """
    household_heads = dict()
    for i, person in popdict.items():
        if person['hhid'] is not None:
            household_heads.setdefault(person['hhid'], np.inf)
            if i < household_heads[person['hhid']]:
                household_heads[person['hhid']] = i  # update the minimum id; synthpops creates the head of the household first for each household so they will have the smallest id of all members in their household

    return household_heads


def get_generated_household_size_distribution(household_sizes):
    """
    Get household size distribution.

    Args:
        household_sizes (dict): size of each generated household

    Returns:
        dict: Dictionary of the generated household size distribution.
    """
    household_size_count = spb.count_values(household_sizes)
    household_size_dist = spb.norm_dic(household_size_count)
    return {k: household_size_dist[k] for k in sorted(household_size_dist.keys())}
