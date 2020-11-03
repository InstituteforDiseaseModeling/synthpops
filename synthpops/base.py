"""
The module contains frequently-used functions that do not neatly fit into other areas of the code base.
"""

# import sciris as sc
import numpy as np
from copy import deepcopy
from . import config as cfg


def norm_dic(dic):
    """
    Normalize the dictionary ``dic``.

    Args:
        dic (dict): A dictionary with numerical values.

    Returns:
        A normalized dictionary.
    """
    # total = np.sum([dic[i] for i in dic], dtype=float)
    total = float(sum(dic.values()))
    if total == 0.0:
        return dic
    return {k: v / total for k, v in dic.items()}


def norm_age_group(age_dic, age_min, age_max):
    """
    Create a normalized dictionary for the range ``age_min`` to ``age_max``, inclusive.

    Args:
        age_dic (dict) : A dictionary with numerical values.
        age_min (int)  : The minimum value of the range for the dictionary.
        age_max (int)  : The maximum value of the range for the dictionary.

    Returns:
        A normalized dictionary for keys in the range ``age_min`` to ``age_max``, inclusive.
    """
    dic = {a: age_dic[a] for a in range(age_min, age_max + 1)}
    return norm_dic(dic)


# Functions related to age distributions
def get_index_by_brackets_dic(brackets):
    """
    Create a dictionary mapping each item in the value arrays to the key. For example, if brackets
    are age brackets, then this function will map each age to the age bracket or bin that it belongs to,
    so that the resulting dictionary will give by_brackets_dic[age_index] = age bracket of age_index.

    Args:
        brackets (dict): A dictionary mapping bracket or bin keys to the array of values that belong to each bracket.

    Returns:
        dict: A dictionary mapping indices to the brackets or bins each index belongs to.

    """
    by_brackets_dic = {}
    for b in brackets:
        for a in brackets[b]:
            by_brackets_dic[a] = b
    return by_brackets_dic


def get_age_by_brackets_dic(age_brackets):
    """
    Create a dictionary mapping age to the age bracket it falls in.

    Args:
        age_brackets (dict): A dictionary mapping age bracket keys to age bracket range.

    Returns:
        A dictionary of age bracket by age.

    Example
    =======

    ::

        age_brackets = sp.get_census_age_brackets(sp.datadir,state_location='Washington',country_location='usa')
        age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)
    """
    return get_index_by_brackets_dic(age_brackets)


def get_ids_by_age_dic(age_by_id_dic):
    """
    Get lists of IDs that map to each age.

    Args:
        age_by_id_dic (dict): A dictionary with the age of each individual by their ID.

    Returns:
        A dictionary listing IDs for each age from a dictionary that maps ID to age.
    """
    max_val = max([v for v in age_by_id_dic.values()])
    ids_by_age_dic = dict.fromkeys(np.arange(max_val + 1))
    for i in ids_by_age_dic:
        ids_by_age_dic[i] = []
    for i in age_by_id_dic:
        ids_by_age_dic[age_by_id_dic[i]].append(i)
    return ids_by_age_dic
