"""Sample distributions, either from real world data or from uniform distributions."""

import os
import numpy as np
import pandas as pd
import sciris as sc
import numba as nb
import random
import itertools
import bisect
from collections import Counter
from . import base as spb
from . import data_distributions as spdata
from .config import logger as log


def set_seed(seed=None):
    """Reset the random seed -- complicated because of Numba."""
    @nb.njit((nb.int64,), cache=True)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed)  # If None, reinitializes it
    if seed is None:  # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed)  # Finally, reset Python's built-in random number generator

    return


def fast_choice(weights):
    """
    Choose an option -- quickly -- from the provided weights. Weights do not need
    to be normalized.

    Reimplementation of random.choices(), removing everything inessential.

    Example:
        fast_choice([0.1,0.2,0.3,0.2,0.1]) # might return 2
    """
    cum_weights = list(itertools.accumulate(weights))
    return bisect.bisect(cum_weights, random.random()*(cum_weights[-1]), 0, len(cum_weights)-1)


# @nb.njit(cache=True)
def sample_single_dict(distr_keys, distr_vals):
    """
    Sample from a distribution.

    Args:
        distr (dict or np.ndarray): distribution

    Returns:
        A single sampled value from a distribution.

    """
    return distr_keys[fast_choice(distr_vals)]
    # sort_inds = np.argsort(distr_keys)
    # sorted_keys = distr_keys[sort_inds]
    # sorted_distr = distr_vals[sort_inds]
    # norm_sorted_distr = np.maximum(0, sorted_distr)  # Don't allow negatives, and mask negative values to 0.

    # sum_norm_sorted_distr = norm_sorted_distr.sum()
    # if sum_norm_sorted_distr > 0:
    #     norm_sorted_distr = norm_sorted_distr / sum_norm_sorted_distr  # Ensure it sums to 1 - normalize all values by the summation, but only if the sum of them is not zero.
    # else:
    #     return 0
    # n = np.random.multinomial(1, norm_sorted_distr, size=1)[0]
    # index = np.where(n)[0][0]
    # return sorted_keys[index]


# @nb.njit((nb.float64[:],), cache=True)
def sample_single_arr(distr):
    """
    Sample from a distribution.

    Args:
        distr (dict or np.ndarray): distribution

    Returns:
        A single sampled value from a distribution.
    """
    if any(distr<0):
        raise Exception('foooo')
    return fast_choice(distr)
    # norm_distr = np.maximum(0, distr)  # Don't allow negatives, and mask negative values to 0.
    # sum_norm_distr = norm_distr.sum()
    # if sum_norm_distr > 0:
    #     norm_distr = norm_distr / sum_norm_distr  # Ensure it sums to 1 - normalize all values by the summation, but only if the sum of them is not zero.
    # else:
    #     return 0
    # n = np.random.multinomial(1, norm_distr, size=1)[0]
    # index = np.where(n)[0][0]
    # return index


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

    age_distr = age_dist_vals[age_min:age_max + 1]  # create an array of the values, not yet normalized
    # norm_age_distr = np.maximum(0, age_distr)  # Don't allow negatives, and mask negative values to 0.
    # sum_norm_age_distr = norm_age_distr.sum()
    # if sum_norm_age_distr > 0:
    #     norm_age_distr = norm_age_distr / (sum_norm_age_distr)  # Ensure it sums to 1 - normalize all values by the summation, but only if the sum of them is not zero.

    age_range = np.arange(age_min, age_max + 1)
    # n = np.random.multinomial(1, norm_age_distr, size=1)[0]
    # index = np.where(n)[0]
    return age_range[fast_choice(age_distr)]


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