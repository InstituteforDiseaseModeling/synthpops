"""
This module contains reading and writing functions for contact networks and populations
"""

import os

import sciris as sc
import numpy as np
import pandas as pd


def write_age_by_uid_dic(datadir, location, state_location, country_location, folder_name, age_by_uid_dic):
    """
    Write the dictionary of ID mapping to age for each individual in the population.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        folder_name (string)      : The name of the folder the location is in, e.g. 'contact_networks'
        age_by_uid_dic (dict)     : A dictionary mapping ID to age for each individual in the population.

    Returns:
        None

    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, folder_name)
    os.makedirs(file_path, exist_ok=True)

    age_by_uid_path = os.path.join(file_path, location + '_' + str(len(age_by_uid_dic)) + '_age_by_uid.dat')

    f_age_uid = open(age_by_uid_path, 'w')

    uids = sorted(age_by_uid_dic.keys())
    for uid in uids:
        f_age_uid.write(str(uid) + ' ' + str(age_by_uid_dic[uid]) + '\n')
    f_age_uid.close()


def read_in_age_by_uid(datadir, location, state_location, country_location, folder_name, n):
    """
    Read dictionary of ID mapping to ages for all individuals from file.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        folder_name (string)      : The name of the folder the location is in, e.g. 'contact_networks'
        n (int)                   : The number of people in the population.

    Returns:
        A dictionary mapping ID to age for all individuals in the population.

    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, folder_name)
    age_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_age_by_uid.dat')
    df = pd.read_csv(age_by_uid_path, header=None, delimiter=' ')
    try:
        return dict(zip(df.iloc[:, 0].values.astype(int), df.iloc[:, 1].values.astype(int)))
    except:
        return dict(zip(df.iloc[:, 0].values, df.iloc[:, 1].values.astype(int)))
    # return sc.objdict(zip(df.iloc[:, 0].values, df.iloc[:, 1].values.astype(int)))


def write_groups_by_age_and_uid(datadir, location, state_location, country_location, folder_name, age_by_uid_dic, group_type, groups_by_uids):
    """
    Write groups to file with both ID and their ages.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state of the location is in.
        country_location (string) : The name of the country the location is in.
        folder_name (string)      : The name of the folder the location is in, e.g. 'contact_networks'
        age_by_uid_dic (dict)     : A dictionary mapping ID to age for each individual in the population.
        group_type (string)       : The name of the group type.
        groups_by_uids (list)     : The list of lists, where each sublist represents a household and the IDs of the household members.

    Returns:
        None

    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, folder_name)
    os.makedirs(file_path, exist_ok=True)

    groups_by_age_path = os.path.join(file_path, location + '_' + str(len(age_by_uid_dic)) + '_synthetic_' + group_type + '_with_ages.dat')
    groups_by_uid_path = os.path.join(file_path, location + '_' + str(len(age_by_uid_dic)) + '_synthetic_' + group_type + '_with_uids.dat')

    fg_age = open(groups_by_age_path, 'w')
    fg_uid = open(groups_by_uid_path, 'w')

    for n, ids in enumerate(groups_by_uids):

        group = groups_by_uids[n]

        for uid in group:

            fg_age.write(str(int(age_by_uid_dic[uid])) + ' ')
            fg_uid.write(str(uid) + ' ')
        fg_age.write('\n')
        fg_uid.write('\n')
    fg_age.close()
    fg_uid.close()


def read_setting_groups(datadir, location, state_location, country_location, folder_name, group_type, n, with_ages=False):
    """
    Read in groups of people interacting in different social settings from file.

    Args:
        datadir (string)          : The file path to the data directory.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        folder_name (string)      : The name of the folder the location is in, e.g. 'contact_networks'
        group_type (string)       : The name of the group type.
        n (int)                   : The number of people in the population.
        with_ages (bool)          : If True, read in the ages of each individual in the group; otherwise, read in their IDs.

    Returns:
        A list of lists where each sublist represents of group of individuals in the same group and thus are contacts of each other.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, folder_name, location + '_' + str(n) + '_synthetic_' + group_type + '_with_uids.dat')
    if with_ages:
        file_path = file_path.replace('_uids', '_ages')
    groups = []
    foo = open(file_path, 'r')
    for c, line in enumerate(foo):
        group = line.strip().split(' ')

        try:
            group = [int(float(i)) for i in group]
        except:
            group = [i for i in group]

        groups.append(group)
    return groups
