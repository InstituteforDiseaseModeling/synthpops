'''
Functions for generating households
'''

import sciris as sc
import numpy as np
import pandas as pd
from collections import Counter
from .config import logger as log, checkmem
from . import base as spb
from . import sampling as spsamp
from . import data_distributions as spdata


class Household(spb.LayerGroup):
    """
    A class for individual households and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the household
    """

    def __init__(self, hhid=None, reference_uid=None, reference_age=None, **kwargs):
        """
        Class constructor for empty household.

        Args:
            **hhid (int)             : household id
            **member_uids (np.array) : ids of household members
            **reference_uid (int)    : id of the reference person
            **reference_age (int)    : age of the reference person
        """
        super().__init__(hhid=hhid, reference_uid=reference_uid, reference_age=reference_age, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a household is valid and update
        to the correct type if necessary.
        """
        super().validate(layer_str='household')
        return

    # To be turned on for vital dynamics...
    # def set_hhid(self, hhid):
    #     """Set the household id."""
    #     self['hhid'] = int(hhid)

    # def set_member_uids(self, member_uids):
    #     """Set the uids of all household members."""
    #     self['member_uids'] =  sc.promotetoarray(member_uids, dtype=int)

    # def set_member_ages(self):
    #     """Set the ages of all household members."""
    #     self['member_ages'] = sc.promotetoarray(member_ages, dtype=int)

    # def set_reference_uid(self, reference_uid):
    #     """Set the uid of the reference person to generate the household members ages."""
    #     self['reference_uid'] = int(reference_uid)

    # def set_reference_age(self):
    #     """Set the age of the reference person to generate the household members ages."""
    #     self['reference_age'] = int(reference_age)


def get_household(pop, hhid):
    """
    Return household with id: hhid.

    Args:
        pop (sp.Pop) : population
        hhid (int)   : household id number

    Returns:
        sp.Household: A populated household.
    """
    if not isinstance(hhid, int):
        raise TypeError(f"hhid must be an int. Instead supplied hhid with type: {type(hhid)}.")
    if len(pop.households) <= hhid:
        raise IndexError(f"Household id (hhid): {hhid} out of range. There are {len(pop.households)} households stored in this object.")
    return pop.households[hhid]


def add_household(pop, household):
    """
    Add a household to the list of households.

    Args:
        pop (sp.Pop)             : population
        household (sp.Household) : household with at minimum the hhid, member_uids, member_ages, reference_uid, and reference_age.
    """
    if not isinstance(household, Household):
        raise ValueError('household is not a sp.Household object.')

    # ensure hhid to match the index in the list
    if household['hhid'] != len(pop.households):
        household['hhid'] = len(pop.households)
    pop.households.append(household)
    pop.n_households = len(pop.households)
    return


def initialize_empty_households(pop, n_households=None):
    """
    Array of empty households.

    Args:
        pop (sp.Pop)       : population
        n_households (int) : the number of households to initialize
    """
    if n_households is not None and isinstance(n_households, int):
        pop.n_households = n_households
    else:
        pop.n_households = 0

    pop.households = [Household() for nh in range(pop.n_households)]
    return


def populate_households(pop, households, age_by_uid):
    """
    Populate all of the households. Store each household at the index corresponding to it's hhid.

    Args:
        pop (sp.Pop)      : population
        households (list) : list of lists where each sublist represents a household and contains the ids of the household members
        age_by_uid (dict) : dictionary mapping each person's id to their age
    """
    # initialize an empty set of households
    # if previously you had 10 households and now you want to repopulate with
    # this method and only supply 5 households, this method will overwrite the list to produce only 5 households
    initialize_empty_households(pop, len(households))

    log.debug("Populating households.")

    # now populate households
    for nh, hh in enumerate(households):
        kwargs = dict(hhid=nh,
                      member_uids=hh,
                      reference_uid=hh[0],  # by default, the reference person is the first in the household in synthpops - with vital dynamics this may change
                      reference_age=age_by_uid[hh[0]]
                      )
        household = Household()
        household.set_layer_group(**kwargs)
        pop.households[household['hhid']] = sc.dcp(household)

    pop.populate = True

    return


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
    age_by_uid = dict()
    homes_by_uids = []

    for h, home in enumerate(homes):

        home_ids = []
        for a in home:
            if use_int:
                uid = len(age_by_uid)
            else:
                uid = sc.uuid(length=id_len)
            age_by_uid[uid] = int(a)
            home_ids.append(uid)

        homes_by_uids.append(home_ids)

    return homes_by_uids, age_by_uid


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
    age_range = np.arange(0, len(age_distr))
    chosen = np.random.choice(age_range, size=n, p=age_distr)
    age_count = Counter(chosen)
    age_count = sc.mergedicts(dict.fromkeys(age_range, 0), age_count)
    return age_count


def generate_living_alone_method_2(hh_sizes, hha_by_size, hha_brackets, age_count):
    """
    Generate the ages of those living alone.

    Args:
        hh_sizes (array)     : The count of household size s at index s-1.
        hha_by_size (matrix) : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)  : The age brackets for the heads of household.
        age_distr (dict)     : The age distribution.

    Returns:
        An array of households of size 1 where each household is a row and the
        value in the row is the age of the household member.
    """
    distr = hha_by_size[0, :]
    distr = distr / np.sum(distr)

    h1_count = hh_sizes[0]
    hha_b = np.random.choice(range(len(distr)), size=h1_count, p=distr)

    hha_b_count = Counter(hha_b)
    hha_living_alone = []
    for hha_bi in hha_brackets:
        possible_hha_bi_ages = []
        for a in hha_brackets[hha_bi]:
            possible_hha_bi_ages.extend([a] * age_count[a])
        np.random.shuffle(possible_hha_bi_ages)
        chosen_hha = possible_hha_bi_ages[0:hha_b_count[hha_bi]]
        hha_living_alone.extend(chosen_hha)
    np.random.shuffle(hha_living_alone)

    homes = np.array(hha_living_alone).astype(int).reshape((len(hha_living_alone), 1))
    return homes


def generate_larger_household_sizes(hh_sizes):
    """
    Create a list of the households larger than 1 in random order so that as
    individuals are placed by age into homes running out of specific ages is not
    systemically an issue for any given household size unless certain sizes
    greatly outnumber households of other sizes.

    Args:
        hh_sizes (array) : The count of household size s at index s-1.

    Returns:
        Np.array: An array of household sizes to be generated and place people
        into households.
    """
    larger_hh_size_array = []
    for hs in range(2, len(hh_sizes) + 1):
        larger_hh_size_array.extend([hs] * hh_sizes[hs - 1])
    larger_hh_size_array = np.array(larger_hh_size_array)
    np.random.shuffle(larger_hh_size_array)
    return larger_hh_size_array


def generate_larger_households_head_ages(larger_hh_size_array, hha_by_size, hha_brackets, ages_left_to_assign):
    """
    Generate the ages of the heads of households for households larger than 2.
    """
    larger_hha_chosen = []

    # go through every household and choose the head age
    for nh, hs in enumerate(larger_hh_size_array):
        hs_distr = hha_by_size[hs - 1, :]
        hbi = spsamp.fast_choice(hs_distr)
        hbi_distr = np.array([ages_left_to_assign[a] for a in hha_brackets[hbi]])

        while sum(hbi_distr) == 0: # pragma: no cover
            hbi = spsamp.fast_choice(hs_distr)
            hbi_distr = np.array([ages_left_to_assign[a] for a in hha_brackets[hbi]])

        hha = hha_brackets[hbi][spsamp.fast_choice(hbi_distr)]
        ages_left_to_assign[hha] -= 1

        larger_hha_chosen.append(hha)

    return larger_hha_chosen, ages_left_to_assign


def generate_larger_households_method_2(larger_hh_size_array, larger_hha_chosen, hha_brackets, cm_age_brackets, cm_age_by_brackets, household_matrix, ages_left_to_assign, homes_dic):
    """
    Assign people to households larger than one person (excluding special
    residences like long term care facilities or agricultural workers living in
    shared residential quarters.

    Args:
        hh_sizes (array)              : The count of household size s at index s-1.
        hha_by_size (matrix)          : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)           : The age brackets for the heads of household.
        cm_age_brackets (dict)        : The age brackets for the contact matrix.
        cm_age_by_brackets (dict)     : A dictionary mapping age to the age bracket range it falls within.
        household_matrix (dict)       : The age-specific contact matrix for the household ontact setting.
        larger_homes_age_count (dict) : Age count of people left to place in households larger than one person.

    Returns:
        dict: A dictionary of households by age indexed by household size.
    """

    # go through every household and assign the ages of the other household members from those left to place
    for nh, hs in enumerate(larger_hh_size_array):

        hha = larger_hha_chosen[nh]
        b = cm_age_by_brackets[hha]

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

    assert sum(ages_left_to_assign.values()) == 0, 'Check failed: generating larger households method 2.'  # at this point everyone should have been placed into a home
    return homes_dic, ages_left_to_assign


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
        dict: Dictionary of the id of the head of the household for each
        household.

    Note:
        In static populations the id of the head of the household is the minimum
        id of the household members. With vital dynamics turned on and
        populations growing or changing households over time, this method will
        need to change and the household head or reference person will need to
        be specified at creation and when those membership events occur.
    """
    household_heads = dict()
    for i, person in popdict.items():
        if person['hhid'] is not None:
            household_heads.setdefault(person['hhid'], np.inf)
            if i < household_heads[person['hhid']]:
                household_heads[person['hhid']] = i  # update the minimum id; synthpops creates the head of the household first for each household so they will have the smallest id of all members in their household

    return household_heads


def get_household_head_ages_by_size(pop):
    """
    Calculate the count of households by size and the age of the head of the
    household, assuming the minimal household members id is the id of the head
    of the household.

    Args:
        pop (sp.Pop) : population object

    Returns:
        np.ndarray: An array with rows as household size and columns as
        household head age brackets.
    """
    popdict = pop.popdict
    loc_pars = sc.dcp(pop.loc_pars)
    loc_pars.location = None
    hha_brackets = spdata.get_head_age_brackets(**loc_pars)  # temporarily location should be None until data json work will automatically search up when data are not available

    # hha_index use age as key and bracket index as value
    hha_index = spb.get_index_by_brackets(hha_brackets)
    uids = get_household_heads(popdict=popdict)
    d = {}
    # construct tables for each houldhold head
    for uid in uids.values():
        d[popdict[uid]['hhid']] = {'hhid': popdict[uid]['hhid'],
                                   'age': popdict[uid]['age'],
                                   'family_size': len(popdict[uid]['contacts']['H']) + 1,
                                   'hh_age_bracket': hha_index[popdict[uid]['age']]}
    df_household_age = pd.DataFrame.from_dict(d, orient="index")
    # aggregate by age_bracket (column) and family_size (row)
    df_household_age = df_household_age.groupby(['hh_age_bracket', 'family_size'], as_index=False).count()\
        .pivot(index='family_size', columns='hh_age_bracket', values='hhid').fillna(0)
    return np.array(df_household_age.values)


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
