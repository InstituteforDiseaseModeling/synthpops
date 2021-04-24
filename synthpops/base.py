"""
The module contains frequently-used functions that do not neatly fit into other areas of the code base.
"""

import numpy as np
import sciris as sc
from collections import Counter
from . import defaults as spd

__all__ = ['LayerGroup']


class LayerGroup(dict):
    """
    A generic class for individual setting group and some methods to operate on each.

    Args:
        kwargs (dict) : data dictionary for the setting group

    Notes:
        Settings currently supported include : households (H), schools (S),
        workplaces (W), and long term care facilities (LTCF).
    """

    def __init__(self, **kwargs):
        """
        Class constructor for an base empty setting group.

        Args:
            **member_uids (np.array) : ids of group members
            **member_ages (np.array) : ages of group members
            **reference_uid (int) : id of the reference person
            **reference_age (int) : age of the reference person
        """
        # set up default values
        default_kwargs = spd.default_layer_info
        kwargs = sc.mergedicts(default_kwargs, kwargs)
        self.update(kwargs)
        self.validate()

        return

    def set_layer(self, **kwargs):
        """"""
        for key, value in kwargs.items():
            self[key] = value
        self.validate()

        return

    def __len__(self):
        """Return the length as the number of members in the layer group"""
        return len(self['member_uids'])

    def validate(self):
        """
        Check that information supplied to make a household is valid and update
        to the correct type if necessary.
        """
        for key in self.keys():
            if key in ['member_uids', 'member_ages']:
                try:
                    self[key] = sc.promotetoarray(self[key], dtype=int)
                except:
                    errmsg = f"Could not convert key {key} to an np.array() with type int. This key only takes arrays with int values."
                    raise TypeError(errmsg)
            else:
                if not isinstance(self[key], int):
                    if self[key] is not None:
                        errmsg = f"Expected type int or None for key {key}. Instead the type of this value is {type(self[key])}."
                        raise TypeError(errmsg)

        return


__all__ += ['norm_dic', 'norm_age_group']


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
__all__ += ['get_index_by_brackets_dic', 'get_age_by_brackets_dic', 'get_ids_by_age_dic']


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


__all__ += ['count_ages', 'get_aggregate_ages', 
            'get_aggregate_matrix', 'get_asymmetric_matrix']


def count_ages(popdict):
    """
    Create an age count from a population dictionary.

    Args:
        popdict (dict): dictionary defining population

    Returns:
        dict: Dictionary of the age count of the population.
    """
    age_count = dict.fromkeys(np.arange(0, spd.settings.max_age), 0)

    for i, person in popdict.items():
        age_count[person['age']] += 1
    return age_count


def get_aggregate_ages(ages, age_by_brackets_dic):
    """
    Create a dictionary of the count of ages by age brackets.

    Args:
        ages (dict)                : A dictionary of age count by single year.
        age_by_brackets_dic (dict) : A dictionary mapping age to the age bracket range it falls within.

    Returns:
        A dictionary of aggregated age count for specified age brackets.

    Example
    =======

    ::

        aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)
        aggregate_matrix = symmetric_matrix.copy()
        aggregate_matrix = sp.get_aggregate_matrix(aggregate_matrix, age_by_brackets_dic)
    """
    bracket_keys = set(age_by_brackets_dic.values())
    aggregate_ages = dict.fromkeys(bracket_keys, 0)
    for a in ages:
        b = age_by_brackets_dic[a]
        aggregate_ages[b] += ages[a]
    return aggregate_ages


def get_aggregate_matrix(matrix, age_by_brackets_dic):
    """
    Aggregate a symmetric matrix to fewer age brackets. Do not use for homogeneous mixing matrix.

    Args:
        matrix (np.ndarray)        : A symmetric age contact matrix.
        age_by_brackets_dic (dict) : A dictionary mapping age to the age bracket range it falls within.

    Returns:
        A symmetric contact matrix (``np.ndarray``) aggregated to age brackets.

    Example
    =======

    ::

        age_brackets = sp.get_census_age_brackets(sp.settings_config.datadir,state_location='Washington',country_location='usa')
        age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

        aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)
        aggregate_matrix = symmetric_matrix.copy()
        aggregate_matrix = sp.get_aggregate_matrix(aggregate_matrix, age_by_brackets_dic)

        asymmetric_matrix = sp.get_asymmetric_matrix(aggregate_matrix, aggregate_age_count)

   """
    n = len(matrix)
    num_agebrackets = len(set(age_by_brackets_dic.values()))
    m_agg = np.zeros((num_agebrackets, num_agebrackets))
    for i in range(n):
        bi = age_by_brackets_dic[i]
        for j in range(n):
            bj = age_by_brackets_dic[j]
            m_agg[bi][bj] += matrix[i][j]
    return m_agg


def get_asymmetric_matrix(symmetric_matrix, aggregate_ages):
    """
    Get the contact matrix for the average individual in each age bracket.

    Args:
        symmetric_matrix (np.ndarray) : A symmetric age contact matrix.
        aggregate_ages (dict)         : A dictionary mapping single year ages to age brackets.

    Returns:
        A contact matrix (``np.ndarray``) whose elements ``M_ij`` describe the contact frequency for the average individual in age bracket ``i`` with all possible contacts in age bracket ``j``.

    Example
    =======

    ::

        age_brackets = sp.get_census_age_brackets(sp.datadir,state_location='Washington',country_location='usa')
        age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

        aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)
        aggregate_matrix = symmetric_matrix.copy()
        aggregate_matrix = sp.get_aggregate_matrix(aggregate_matrix, age_by_brackets_dic)

        asymmetric_matrix = sp.get_asymmetric_matrix(aggregate_matrix, aggregate_age_count)
    """
    M = sc.dcp(symmetric_matrix)
    for a in aggregate_ages:
        M[a, :] = M[a, :] / float(aggregate_ages[a])

    return M


__all__ += ['get_bin_edges', 'get_bin_labels',
            'count_values', 'count_binned_values', 'binned_values_dist']


def get_bin_edges(size_brackets):
    """
    Get the bin edges for size brackets.

    Args:
        size_brackets (dict): dictionary mapping bracket or bin number to an array of the range of sizes

    Returns:
        An array of the bin edges.
    """

    return np.array([size_brackets[0][0]] + [size_brackets[b][-1] + 1 for b in sorted(size_brackets.keys())])


def get_bin_labels(size_brackets):
    """
    Get the bin labels from the values contained within each bracket or bin.

    Args:
        size_brackets (dict): dictionary mapping bracket or bin number to an array of the range of sizes

    Returns:
        A list of bin labels.
    """
    return [f"{size_brackets[b][0]}-{size_brackets[b][-1]}" for b in size_brackets]


def count_values(dic):
    """
    Counter of values in the dictionary. Keys in the returned dictionary are values from the input dictionary.

    Args:
        dic (dict) : dictionary with sortable values

    Returns:
        dict: Dictionary of the count of values.
    """
    value_count = Counter(dic.values())
    return {k: value_count[k] for k in sorted(value_count.keys())}


def count_binned_values(dic, bins=None):
    """
    Binned counter of values in the dictionary. Indices are the bin indices from the input bins.

    Args:
        dic (dict)   : dictionary with sortable and binnable values
        bins (array) : array of bin edges

    Returns:
        array: Array of the count of values binned
    """
    values = list(dic.values())
    hist, bins = np.histogram(values, bins=bins, density=0)
    return hist, bins


def binned_values_dist(dic, bins=None):
    """
    Binned distribution of values in the dictionary. Indices are the bin indices from the input bins.

    Args:
        dic (dict)   : dictionary with sortable and binnable values
        bins (array) : array of bin edges

    Returns:
        array: Array of the binned distribution of values.
    """
    hist, bins = count_binned_values(dic, bins)
    if sum(hist) > 0:
        dist = hist / sum(hist)
    else:
        dist = hist
    return dist