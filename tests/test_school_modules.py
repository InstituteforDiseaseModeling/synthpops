import os
from copy import deepcopy
from collections import Counter

import sciris as sc
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean

# from . import base as spb
# from . import data_distributions as spdata
# from . import sampling as spsamp
# from . import contacts as spct
import synthpops as sp


def generate_random_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, verbose=False):
    """
    Args:
        syn_school_uids (list): list of uids of students in the school
        syn_school_ages (list): list of the ages of the students in the school
        age_by_uid_dic (dict): dict mapping uid to age
        grade_age_mapping (dict): dict mapping grade to an age
        age_grade_mapping (dict): dict mapping age to a grade
        average_class_size (int): average class size
        inter_grade_mixing (float): percent of within grade edges that rewired to create edges across grades

    Returns:
        A edges between students in school.

    """

    # what are the ages in the school
    age_counter = Counter(syn_school_ages)
    age_keys = sorted(age_counter.keys())
    age_keys_indices = {a: i for i, a in enumerate(age_keys)}

    # create a dictionary with the list of uids for each age/grade
    uids_in_school_by_age = {}
    for a in age_keys:
        uids_in_school_by_age[a] = []

    for uid in syn_school_uids:
        a = age_by_uid_dic[uid]
        uids_in_school_by_age[a].append(uid)

    # create a graph of contacts in the school
    G = nx.Graph()

    # for a in grouped_inschool_ids:
    for a in uids_in_school_by_age:

        p = float(average_class_size)/len(uids_in_school_by_age[a])  # density of contacts within each grade
        Ga = nx.erdos_renyi_graph(len(uids_in_school_by_age[a]), p)  # creates a well mixed graph across the grade/age
        for e in Ga.edges():
            i, j = e

            # add each edge to the overall school graph
            G.add_edge(uids_in_school_by_age[a][i], uids_in_school_by_age[a][j])

    if verbose:
        print('clustering within the school', nx.transitivity(G))

    # rewire some edges between people within the same grade/age to now being edges across grades/ages
    E = G.edges()
    E = [e for e in E]
    np.random.shuffle(E)

    nE = int(len(E)/2.)  # we'll loop over edges in pairs so only need to loop over half the length

    for n in range(nE):
        if np.random.binomial(1, p=inter_grade_mixing):

            i = 2 * n
            j = 2 * n + 1

            ei = E[i]
            ej = E[j]

            ei1, ei2 = ei
            ej1, ej2 = ej

            # try to switch from ei1-ei2, ej1-ej2 to ei1-ej2, ej1-ei2
            if ei1 != ej1 and ei2 != ej2 and ei1 != ej2 and ej1 != ei2:
                new_ei = (ei1, ej2)
                new_ej = (ei2, ej1)

            # instead try to switch from ei1-ei2, ej1-ej2 to ei1-ej1, ei2-ej2
            elif ei1 != ej2 and ei2 != ej1 and ei1 != ej1 and ej2 != ei2:
                new_ei = (ei1, ej1)
                new_ej = (ei2, ej2)

            else:
                continue

            G.remove_edges_from([ei, ej])
            G.add_edges_from([new_ei, new_ej])

    # calculate school age mixing
    ecount = np.zeros((len(age_keys), len(age_keys)))
    for e in G.edges():
        i, j = e

        age_i = age_by_uid_dic[i]
        index_i = age_keys_indices[age_i]
        age_j = age_by_uid_dic[j]
        index_j = age_keys_indices[age_j]

        ecount[index_i][index_j] += 1
        ecount[index_j][index_i] += 1

    if verbose:
        print('within school age mixing matrix', ecount)

    E = [e for e in G.edges()]

    return E


def generate_clustered_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, return_edges=False, verbose=False):
    """
    Args:
        syn_school_uids (list): list of uids of students in the school
        syn_school_ages (list): list of the ages of the students in the school
        age_by_uid_dic (dict): dict mapping uid to age
        grade_age_mapping (dict): dict mapping grade to an age
        age_grade_mapping (dict): dict mapping age to a grade
        average_class_size (int): average class size
        inter_grade_mixing (float): percent of within grade edges that rewired to create edges across grades

    Returns:
        A edges between students in school.

    """

    # what are the ages in the school
    age_counter = Counter(syn_school_ages)
    age_keys = sorted(age_counter.keys())
    age_keys_indices = {a: i for i, a in enumerate(age_keys)}

    # create a dictionary with the list of uids for each age/grade
    uids_in_school_by_age = {}
    for a in age_keys:
        uids_in_school_by_age[a] = []

    for uid in syn_school_uids:
        a = age_by_uid_dic[uid]
        uids_in_school_by_age[a].append(uid)

    G = nx.Graph()

    nodes_left = []

    groups = []

    for a in uids_in_school_by_age:
        nodes = deepcopy(uids_in_school_by_age[a])
        np.random.shuffle(nodes)

        ln = age_counter[a]

        while len(nodes) > 0:
            cluster_size = np.random.poisson(average_class_size)

            if cluster_size > len(nodes):
                nodes_left += list(nodes)
                break

            group = nodes[:cluster_size]
            groups.append(group)

            nodes = nodes[cluster_size:]

            if return_edges:
                Gn = nx.complete_graph(cluster_size)
                for e in Gn.edges():
                    i, j = e
                    node_i = group[i]
                    node_j = group[j]
                    G.add_edge(node_i, node_j)

    np.random.shuffle(nodes_left)
    
    while len(nodes_left) > 0:
        cluster_size = np.random.poisson(average_class_size)
        
        if cluster_size > len(nodes_left):
            cluster_size = len(nodes_left)
            




def add_contacts_from_edgelist(popdict, edgelist, setting):

    for e in edgelist:
        i, j = e

        popdict[i]['contacts'][setting].add(j)
        popdict[j]['contacts'][setting].add(i)

    return popdict



# def add_teachers_to_random_classes_by_grade()


grade_age_mapping = {i: i+5 for i in range(13)}
age_grade_mapping = {i+5: i for i in range(13)}

syn_school_ages = [5, 6, 8, 5, 9, 7, 8, 9, 5, 6, 7, 8, 8, 9, 9, 5, 6, 7, 8, 9, 5, 6, 8, 9, 9, 5, 6, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7,
                   6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 
                   ]


syn_school_uids = np.random.choice(np.arange(250), replace=False, size=len(syn_school_ages))
print(syn_school_uids)


age_by_uid_dic = {}

for n in range(len(syn_school_uids)):
    uid = syn_school_uids[n]
    a = syn_school_ages[n]
    age_by_uid_dic[uid] = a

average_class_size = 20
inter_grade_mixing = 0.1


generate_random_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing)
