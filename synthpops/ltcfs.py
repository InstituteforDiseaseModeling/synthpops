"""
Modeling Seattle Metro Long Term Care Facilities

"""

import numpy as np

import os
import math
from copy import deepcopy

from . import base as spb
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contact_networks as spcnx
from . import school_modules as spsm
from .config import logger as log
from . import config as cfg

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