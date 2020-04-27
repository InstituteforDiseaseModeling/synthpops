import numpy as np
from copy import deepcopy
from . import synthpops as sp
from .config import datadir


def norm_dic(dic):
    """
    Return normalized dict.
    """
    total = np.sum([dic[i] for i in dic], dtype=float)
    if total == 0.0:
        return dic
    new_dic = {}
    for i in dic:
        new_dic[i] = float(dic[i])/total
    return new_dic


def norm_age_group(age_dic, age_min, age_max):
    dic = {}
    for a in range(age_min, age_max+1):
        dic[a] = age_dic[a]
    return norm_dic(dic)


# Functions related to age distributions

def get_age_by_brackets_dic(age_brackets):
    """
    Returns dict of age bracket by age.
    """
    age_by_brackets_dic = {}
    for b in age_brackets:
        for a in age_brackets[b]:
            age_by_brackets_dic[a] = b
    return age_by_brackets_dic


def get_aggregate_ages(ages, age_by_brackets_dic):
    """
    Return an aggregate age count for specified age brackets (values in age_by_brackets_dic)
    """
    bracket_keys = set(age_by_brackets_dic.values())
    aggregate_ages = dict.fromkeys(bracket_keys, 0)
    for a in ages:
        b = age_by_brackets_dic[a]
        aggregate_ages[b] += ages[a]
    return aggregate_ages


def get_aggregate_age_dict_conversion(larger_aggregate_ages, larger_age_brackets, smaller_age_brackets, age_by_brackets_dic_larger, age_by_brackets_dic_smaller):
    """
    Convert the aggregate age count in larger_aggregate_ages from a larger number of age brackets to a smaller number of age brackets
    """
    if len(larger_age_brackets) < len(smaller_age_brackets): raise NotImplementedError('Cannot reduce aggregate ages any further.')
    smaller_aggregate_ages = dict.fromkeys(smaller_age_brackets.keys(), 0)
    for lb in larger_age_brackets:
        a = larger_age_brackets[lb][0]
        sb = age_by_brackets_dic_smaller[a]
        smaller_aggregate_ages[sb] += larger_aggregate_ages[lb]
    return smaller_aggregate_ages


# Functions related to contact matrices

def get_aggregate_matrix(M, age_by_brackets_dic):
    """
    Return symmetric contact matrix aggregated to age brackets. Do not use for community (homogeneous) mixing matrix
   """
    N = len(M)
    num_agebrackets = len(set(age_by_brackets_dic.values()))
    M_agg = np.zeros((num_agebrackets, num_agebrackets))
    for i in range(N):
        bi = age_by_brackets_dic[i]
        for j in range(N):
            bj = age_by_brackets_dic[j]
            M_agg[bi][bj] += M[i][j]
    return M_agg


def get_asymmetric_matrix(symmetric_matrix, aggregate_ages):
    """
    Return asymmetric contact matrix from symmetric contact matrix. Now the element M_ij represents the number of contacts of age group j for the average individual of age group i.
    """
    M = deepcopy(symmetric_matrix)
    for a in aggregate_ages:
        M[a, :] = M[a, :]/float(aggregate_ages[a])

    return M


def get_symmetric_community_matrix(ages):
    """
    Return symmetric homogeneous community matrix for age count in ages.
    """
    N = len(ages)
    M = np.ones((N, N))
    for a in range(N):
        M[a, :] = M[a, :] * ages[a]
        M[:, a] = M[:, a] * ages[a]
    for a in range(N):
        M[a, a] -= ages[a]
    M = M/(np.sum([ages[a] for a in ages], dtype=float) - 1)
    return M


def combine_matrices(matrix_dic, weights_dic, num_agebrackets):
    """
    Returns a contact matrix that is a linear combination of setting specific matrices given weights for each setting.
    """
    M = np.zeros((num_agebrackets, num_agebrackets))
    for setting_code in weights_dic:
        M += matrix_dic[setting_code] * weights_dic[setting_code]
    return M


def get_ids_by_age_dic(age_by_id_dic):
    """
    Returns a dictionary listing out ids for each age from a dictionary that maps id to age.
    """
    max_val = max([v for v in age_by_id_dic.values()])
    ids_by_age_dic = dict.fromkeys(np.arange(max_val+1))
    for i in ids_by_age_dic:
        ids_by_age_dic[i] = []
    for i in age_by_id_dic:
        ids_by_age_dic[age_by_id_dic[i]].append(i)
    return ids_by_age_dic


def get_uids_by_age_dic(popdict):
    """
    Returns a dictionary listing out uids for each age from a dictionary that maps uid to age.
    """
    uids_by_age_dic = {}
    for uid in popdict:
        uids_by_age_dic.setdefault(popdict[uid]['age'], [])
        uids_by_age_dic[popdict[uid]['age']].append(uid)
    return uids_by_age_dic
