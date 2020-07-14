"""
Sample distributions, either from real world data or from uniform distributions
"""

import os
import numpy as np
import pandas as pd
import sciris as sc
import numba as nb
import random
from collections import Counter
from . import base as spb
from . import data_distributions as spdata



def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit((nb.int64,), cache=True)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed) # If None, reinitializes it
    if seed is None: # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed) # Finally, reset Python's built-in random number generator

    return

# @nb.njit((nb.int64[:], nb.float64[:]))
def sample_single_dict(distr_keys, distr_vals):
    """
    Sample from a distribution.

    Args:
        distr (dict or np.ndarray): distribution

    Returns:
        A single sampled value from a distribution.
    """
    sort_inds = np.argsort(distr_keys)
    sorted_keys = distr_keys[sort_inds]
    sorted_distr = distr_vals[sort_inds]
    norm_sorted_distr = np.maximum(0, sorted_distr)  # Don't allow negatives, and mask negative values to 0.

    eps = 1e-9  # This is required with Numba to avoid "E   ValueError: binomial(): p outside of [0, 1]" errors for some reason
    if norm_sorted_distr.sum() > 0:
        norm_sorted_distr = norm_sorted_distr/(eps+norm_sorted_distr.sum())  # Ensure it sums to 1 - normalize all values by the summation, but only if the sum of them is not zero.
    else:
        return 0
    n = np.random.multinomial(1, norm_sorted_distr, size=1)[0]
    index = np.where(n)[0][0]
    return sorted_keys[index]


# @nb.njit((nb.float64[:],), cache=True)
def sample_single_arr(distr):
    """
    Sample from a distribution.

    Args:
        distr (dict or np.ndarray): distribution

    Returns:
        A single sampled value from a distribution.
    """
    eps = 1e-9  # This is required with Numba to avoid "E   ValueError: binomial(): p outside of [0, 1]" errors for some reason
    norm_distr = np.maximum(0, distr)  # Don't allow negatives, and mask negative values to 0.
    if norm_distr.sum() > 0:
        norm_distr = norm_distr/(eps+norm_distr.sum())  # Ensure it sums to 1 - normalize all values by the summation, but only if the sum of them is not zero.
    else:
        return 0
    n = np.random.multinomial(1, norm_distr, size=1)[0]
    index = np.where(n)[0][0]
    return index


# @nb.njit((nb.float64[:], nb.int64), cache=True)
def resample_age(age_dist_vals, age):
    """
    Resample age from single year age distribution.

    Args:
        single_year_age_distr (arr) : age distribution, ordered by age
        age (int)                   : age as an integer
    Returns:
        Resampled age as an integer.
    """
    if age == 0:
        age_min = 0
        age_max = 1
    elif age == 1:
        age_min = 0
        age_max = 2
    elif age >= 2 and age <= 98:
        age_min = age - 2
        age_max = age + 2
    elif age == 99:
        age_min = 97
        age_max = 99
    else:
        age_min = 98
        age_max = 100

    age_distr = age_dist_vals[age_min:age_max+1]  # create an array of the values, not yet normalized
    norm_age_distr = np.maximum(0, age_distr)  # Don't allow negatives, and mask negative values to 0.
    if norm_age_distr.sum() > 0:
        norm_age_distr = norm_age_distr/norm_age_distr.sum()  # Ensure it sums to 1 - normalize all values by the summation, but only if the sum of them is not zero.
    age_range = np.arange(age_min, age_max+1)
    n = np.random.multinomial(1, norm_age_distr, size=1)[0]
    index = np.where(n)[0]
    return age_range[index][0]


def sample_from_range(distr, min_val, max_val):
    """
    Sample from a distribution from min_val to max_val, inclusive.

    Args:
        distr (dict)  : distribution with integer keys
        min_val (int) : minimum of the range to sample from
        max_val (int) : maximum of the range to sample from
    Returns:
        A sampled number from the range min_val to max_val in the distribution distr.
    """
    new_distr = spb.norm_age_group(distr, min_val, max_val)
    distr_keys = np.array(list(new_distr.keys()), dtype=np.int64)
    distr_vals = np.array(list(new_distr.values()), dtype=np.float64)
    return sample_single_dict(distr_keys, distr_vals)


def sample_bracket(distr, brackets):
    """
    Sample bracket from a distribution (potentially absolete).

    Args:
        distr (dict or np.ndarray): distribution for bracket keys

    Returns:
        A sampled bracket from a distribution.
    """
    sorted_keys = sorted(distr.keys())
    sorted_distr = [distr[k] for k in sorted_keys]
    n = np.random.multinomial(1, sorted_distr, size=1)[0]
    index = np.where(n)[0][0]
    return index


def sample_n(nk, distr):
    """
    Sample nk values from a distribution

    Args:
        nk (int)                   : number of samples
        distr (dict or np.ndarray) : distribution

    Returns:
        A dictionary with the count for n samples from a distribution
    """
    if type(distr) == dict:
        distr = spb.norm_dic(distr)
        sorted_keys = sorted(distr.keys())
        sorted_distr = [distr[k] for k in sorted_keys]
        n = np.random.multinomial(nk, sorted_distr, size=1)[0]
        dic = dict(zip(sorted_keys, n))
        return dic
    elif type(distr) == np.ndarray:
        distr = distr / np.sum(distr)
        n = np.random.multinomial(nk, distr, size=1)[0]
        dic = dict(zip(np.arange(len(distr)), n))
        return dic


def sample_contact_age(age, age_brackets, age_by_brackets_dic, age_mixing_matrix, single_year_age_distr=None):
    """
    Sample the age of a contact from age mixing patterns. Age of contact is uniformly drawn from the age bracket sampled from the age mixing matrix, unless single_year_age_distr is available.

    Args:
        age (int)                    : age of reference individual
        age_brackets (dict)          : dictionary mapping age bracket keys to age bracket range
        age_mixing_matrix (matrix)   : age specific contact matrix
        single_year_age_distr (dict) : age distribution by single year ages if available

    Returns:
        Age of contact by age of individual sampled from an age mixing matrix.

    """
    b = age_by_brackets_dic[age]
    b = min(b, age_mixing_matrix.shape[0]-1) # Ensure it doesn't go past the end of the array
    b_contact = sample_single_arr(age_mixing_matrix[b, :])
    if single_year_age_distr is None:
        a = np.random.choice(age_brackets[b_contact])
    else:
        a = sample_from_range(single_year_age_distr, age_brackets[b_contact][0], age_brackets[b_contact][-1])

    return a


def sample_n_contact_ages(n_contacts, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic, weights_dic, single_year_age_distr=None):
    """
    Sample the age of n_contacts contacts from age mixing patterns. Age of each contact is uniformly drawn from the age bracket sampled from the age
    mixing matrix, unless single_year_age_distr is available. Combines setting specific weights to create an age mixing matrix
    from which contact ages are sampled.

    Args:
        n_contacts(int)              : number of contacts to draw ages for
        age (int)                    : age of reference individual
        age_brackets (dict)          : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)   : dictionary mapping age to the age bracket range it falls in
        age_mixing_matrix_dic (dict) : dictionary of age specific contact matrix for different physical contact settings
        weights_dic (dict)           : weights to combine contact matrices
        single_year_age_distr (dict) : age distribution by single year ages if available

    Returns:
        List of ages of n_contacts contacts by age of individual sampled from a combined age mixing matrix.

    """
    num_agebrackets = len(age_brackets)
    age_mixing_matrix = spb.combine_matrices(age_mixing_matrix_dic, weights_dic, num_agebrackets)
    contact_ages = []
    for i in range(n_contacts):
        contact_ages.append(sample_contact_age(age, age_brackets, age_by_brackets_dic, age_mixing_matrix, single_year_age_distr))
    return contact_ages


def sample_n_contact_ages_with_matrix(n_contacts, age, age_brackets, age_by_brackets_dic, age_mixing_matrix, single_year_age_distr=None):
    """
    Sample the age of n_contacts contacts from age mixing matrix. Age of each contact is uniformly drawn from the age bracket sampled from the age
    mixing matrix, unless single_year_age_distr is available.

    Args:
        n_contacts(int)              : number of contacts to draw ages for
        age (int)                    : age of reference individual
        age_brackets (dict)          : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)   : dictionary mapping age to the age bracket range it falls in
        age_mixing_matrix (matrix)   : age specific contact matrix
        weights_dic (dict)           : weights to combine contact matrices
        single_year_age_distr (dict) : age distribution by single year ages if available

    Returns:
        List of ages of n_contacts contacts by age of individual sampled from an age mixing matrix.

    """
    contact_ages = []
    for i in range(n_contacts):
        contact_ages.append(sample_contact_age(age, age_brackets, age_by_brackets_dic, age_mixing_matrix, single_year_age_distr))
    return contact_ages


def get_n_contact_ids_by_age(contact_ids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic):
    """
    Get ids for the contacts with ages in contact_ages.

    Args:
        contact_ids_by_age_dic (dict): dictionary mapping lists of ids to the age of individuals with those ids
        contact_ages (list)          : list of integer ages
        age_brackets (dict)          : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)   : dictionary mapping age to the age bracket range it falls in

    Return set of ids of n_contacts sampled from an age mixing matrix, where potential contacts are chosen from a list of contact ids by age
    """
    contact_ids = set()
    for contact_age in contact_ages:
        age_list = sorted(list(contact_ids_by_age_dic.keys()))
        ind = sc.findnearest(age_list, contact_age)
        these_ids = contact_ids_by_age_dic[age_list[ind]]
        if len(these_ids) > 0:
            contact_id = np.random.choice(these_ids)
        else:
            b_contact = age_by_brackets_dic[contact_age]
            potential_contacts = []
            for a in age_brackets[b_contact]:
                potential_contacts += contact_ids_by_age_dic[a]
            contact_id = np.random.choice(potential_contacts)
        contact_ids.add(contact_id)
    return contact_ids


@nb.njit((nb.int64,), cache=True)
def pt(rate):
    '''
    Results of a Poisson trial

    Args:
        rate (float): Poisson rate

    Returns result (bool) of Poisson trial.
    '''
    return np.random.poisson(rate, 1)[0]


def get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets, min_age=0, max_age=100, age_mean=40, age_std=20):
    '''
    Sample a person's age and sex based on gender and age census data defined for age brackets. Else, return random age and sex.

    Args:
        gender_fraction_by_age (dict): dictionary of the fractions for two genders by age bracket
        age_bracket_distr (dict):    : distribution of ages by brackets
        age_brackets (dict)          : dictionary mapping age bracket keys to age bracket range
        min_age (int)                : minimum age to draw
        max_age (int)                : maximum age to draw
        age_mean (int)               : mean of age distribution
        age_std (int)                : standard deviation of age distribution

    Returns:
        Sampled age (float), sex (int; 0 for female, 1 for male)
    '''
    try:
        b = sample_bracket(age_bracket_distr, age_brackets)
        age = np.random.choice(age_brackets[b])
        sex = np.random.binomial(1, gender_fraction_by_age['male'][b])
        return age, sex
    except:
        sex = np.random.randint(2)  # Define female (0) or male (1) -- evenly distributed
        age = np.random.normal(age_mean, age_std)  # Define age distribution for the crew and guests
        age = np.median([min_age, age, max_age])  # Bound age by the interval
        return age, sex


def get_age_sex_n(gender_fraction_by_age, age_bracket_distr, age_brackets, n_people=1, min_age=0, max_age=100):
    """
    Sample n_people peoples' age and sex from gender and age census data defined for age brackets. Else, return random ages and sex.
    Two lists ordered by age bracket so that people from the first age bracket show up at the front of both lists and people from the last age bracket show up at the end.

    Args:
        gender_fraction_by_age (dict): dictionary of the fractions for two genders by age bracket
        age_bracket_distr (dict):    : distribution of ages by brackets
        age_brackets (dict)          : dictionary mapping age bracket keys to age bracket range
        n_people (int)               : number of people to draw age and sex for
        min_age (int)                : minimum age to draw
        max_age (int)                : maximum age to draw
        age_mean (int)               : mean of age distribution
        age_std (int)                : standard deviation of age distribution

    Returns:
        Two lists of sampled ages (float) and sexes (int; 0 for female, 1 for male) ordered by age bracket so that people from the
        first age bracket show up at the front of both lists and people from the last age bracket show up at the end.

    """
    n_people = int(n_people)

    if age_bracket_distr is None:
        sexes = np.random.binomial(1, p=0.5, size=n_people)
        ages = np.random.randint(min_age, max_age+1, size=n_people)  # should return a flat distribution if we don't know the age distribution, not a normal distribution...
        ages = [np.median([min_age, int(a), max_age]) for a in ages]

    else:
        bracket_count = sample_n(n_people, age_bracket_distr)
        ages, sexes = [], []

        for b in bracket_count:
            sex_probabilities = [gender_fraction_by_age['female'][b], gender_fraction_by_age['male'][b]]
            ages_in_bracket = np.random.choice(age_brackets[b], bracket_count[b])
            sexes_in_bracket = np.random.choice(np.arange(2), bracket_count[b], p=sex_probabilities)
            ages += list(ages_in_bracket)
            sexes += list(sexes_in_bracket)

    return ages, sexes


def get_seattle_age_sex(datadir, location='seattle_metro', state_location='Washington', country_location='usa'):
    '''
    Sample a person's age and sex based on US gender and age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        Sampled age (float), sex (int; 0 for female, 1 for male)

    '''
    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location, state_location, country_location)
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location)

    age, sex = get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets)
    return age, sex


def get_seattle_age_sex_n(datadir, location='seattle_metro', state_location='Washington', country_location='usa', n_people=1e4):
    '''
    Sample n_people peoples' age and sex based on US gender and age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        n_people (float or int)   : number of people to draw age and sex for

    Returns:
        Two lists of sampled ages (float) and sexes (int; 0 for female, 1 for male) ordered by age bracket so that people from the
        first age bracket show up at the front of both lists and people from the last age bracket show up at the end.

    '''
    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location, state_location, country_location)
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location)

    ages, sexes = get_age_sex_n(gender_fraction_by_age, age_bracket_distr, age_brackets, n_people)
    return ages, sexes


def get_usa_age_sex(datadir, location='seattle_metro', state_location='Washington', country_location='usa'):
    '''
    Sample a person's age and sex based on US gender and age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        Sampled age (float), sex (int; 0 for female, 1 for male)

    '''
    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location, state_location, country_location)
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location)

    age, sex = get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets)
    return age, sex


def get_usa_age_sex_n(datadir, location='seattle_metro', state_location='Washington', country_location='usa', n_people=1e4):
    """
    Sample n_people peoples' age and sex based on US gender and age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        n_people (float or int)   : number of people to draw age and sex for

    Returns:
        Two lists of sampled ages (float) and sexes (int; 0 for female, 1 for male) ordered by age bracket so that people from the
        first age bracket show up at the front of both lists and people from the last age bracket show up at the end.

    """
    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location, state_location, country_location)
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location)

    ages, sexes = get_age_sex_n(gender_fraction_by_age, age_bracket_distr, age_brackets, n_people)
    return ages, sexes


def get_usa_age_n(datadir, sexes, location='seattle_metro', state_location='Washington', country_location='usa'):
    """
    Sample n_people peoples' age based on list of sexes supplied and US gender and age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)          : file path to the data directory
        sexes (list)              : list of sexes
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        n_people (float or int)   : number of people to draw age and sex for

    Returns:
        Two lists of sampled ages (float) and sexes (int; 0 for female, 1 for male) ordered by age bracket so that people from the
        first age bracket show up at the front of both lists and people from the last age bracket show up at the end.

    """
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location)

    sex_count = Counter(sexes)
    sex_age_distr = {0: gender_fraction_by_age['female'], 1: gender_fraction_by_age['male']}

    ages, sexes = [], []

    for sex in sex_count:
        bracket_count = sample_n(sex_count[sex], sex_age_distr[sex])
        for b in bracket_count:
            ages_in_bracket = np.random.choice(age_brackets[b], bracket_count[b])
            ages += list(ages_in_bracket)
        sexes += [sex] * sex_count[sex]

    return ages, sexes


def get_usa_sex_n(datadir, ages, location='seattle_metro', state_location='Washington', country_location='usa'):
    """
    Sample n_people peoples' sex based on list of ages supplied and US gender and age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)          : file path to the data directory
        ages (list)               : list of ages
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        n_people (float or int)   : number of people to draw age and sex for

    Returns:
        Two lists of sampled ages (float) and sexes (int; 0 for female, 1 for male) ordered by age bracket so that people from the
        first age bracket show up at the front of both lists and people from the last age bracket show up at the end.

    """

    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    age_count = Counter(ages)
    bracket_count = spb.get_aggregate_ages(age_count, age_by_brackets_dic)

    ages, sexes = [], []

    for b in bracket_count:

        # sex_probabilities = [gender_fraction_by_age['female'][b], gender_fraction_by_age['male'][b]]
        sexes_in_bracket = np.random.binomial(1, p=gender_fraction_by_age['female'][b], size=bracket_count[b])
        # sexes_in_bracket = np.random.choice(np.arange(2), bracket_count[b], sex_probabilities)
        ages_in_bracket = []
        for a in age_brackets[b]:
            ages_in_bracket += [a] * age_count[a]
        ages += ages_in_bracket
        sexes += list(sexes_in_bracket)

    return ages, sexes


def get_age_n(datadir, n, location='seattle_metro', state_location='Washington', country_location='usa', age_brackets_file=None, age_bracket_distr_file=None, age_brackets=None, age_bracket_distr=None):
    """
    Sample n_people peoples' age based on age census data defined for age brackets, with defaults set to Seattle, Washington.

    Args:
        datadir (string)                : file path to the data directory
        n (float or int)                : number of people to draw age and sex for
        location (string)               : name of the location
        state_location (string)         : name of the state the location is in
        country_location (string)       : name of the country the location is in
        age_brackets_file (string)      : user file path to get age brackets from
        age_bracket_distr_file (string) : user file path to get age distribution by brackets from
        age_brackets (dict)             : dictionary mapping age bracket keys to age bracket range
        age_bracket_distr (dict)        : : distribution of ages by brackets

    Returns:
        List of sampled ages (float) ordered by age bracket so that people from the first age bracket show up at the front of the list and people from the last age bracket show up at the end.

    """
    if age_brackets is None:
        age_brackets = spdata.get_census_age_brackets(datadir, state_location, country_location, age_brackets_file)
    if age_bracket_distr is None:
        age_bracket_distr = spdata.read_age_bracket_distr(datadir, location, state_location, country_location, age_bracket_distr_file)

    # check the number of age brackets match
    if len(age_brackets) != len(age_bracket_distr):
        raise Exception("age_brackets and age_bracket_distr don't match in length. Try again.")

    ages = []
    bracket_count = sample_n(n, age_bracket_distr)
    for b in bracket_count:
        ages_in_bracket = np.random.choice(age_brackets[b], bracket_count[b])
        ages += list(ages_in_bracket)

    return ages


def get_mortality_rates_filepath(path):
    """
    Get file path to mortality rates.

    Args:
        path (string): path to folder containing mortality rates by age brackets data.

    Returns:
        The filepath to mortality rates by age bracket.
    """
    return os.path.join(path, 'mortality_rates_by_age_bracket.dat')


def get_mortality_rates_by_age_bracket(file_path):
    """
    Get mortality rates by age bracket

    Args:
        file_path (string): path to mortality rates by age bracket data.

    Returns:
        A dictionary of mortality rates by age bracket.

    """
    df = pd.read_csv(file_path)
    return dict(zip(df.age_bracket, df.rate))


def get_mortality_rates_by_age(mortality_rate_by_age_bracket, mortality_age_brackets):
    """
    Get mortality rates by age

    Args:
        mortality_rate_by_age_bracket (dict) : dictionary of mortality rates by age bracket
        mortality_age_brackets (dict)        : dictionary of age brackets for raw mortality rate data

    Returns:
        A dictionary of mortality rates by age.
    """
    mortality_rates = {}
    for b in mortality_rate_by_age_bracket:
        for a in mortality_age_brackets[b]:
            mortality_rates[a] = mortality_rate_by_age_bracket[b]
    return mortality_rates


def calc_death(person_age, mortality_rates):
    """
    Binomial draw of whether or not an individual succumbs to disease.

    Args:
        person_age (int): age of the ill individual.
        mortality_rates (dict): dictionary of mortality rates by age
    Returns:
        Bool representing the results of a binomial test; 1 for death, 0 for staying alive.

    """
    return np.random.binomial(1, mortality_rates[person_age])
