import os
from copy import deepcopy
from collections import Counter
from itertools import combinations

import sciris as sc
import numpy as np
import pandas as pd
import networkx as nx
import math

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
    E = [e for e in G.edges()]
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
    if verbose:
        ecount = np.zeros((len(age_keys), len(age_keys)))
        for e in G.edges():
            i, j = e

            age_i = age_by_uid_dic[i]
            index_i = age_keys_indices[age_i]
            age_j = age_by_uid_dic[j]
            index_j = age_keys_indices[age_j]

            ecount[index_i][index_j] += 1
            ecount[index_j][index_i] += 1

        print('within school age mixing matrix')
        print(ecount)

    return list(G.edges())


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

        while len(nodes) > 0:
            cluster_size = np.random.poisson(average_class_size)

            if cluster_size > len(nodes):
                nodes_left += list(nodes)
                break

            group = nodes[:cluster_size]
            groups.append(group)

            nodes = nodes[cluster_size:]

    np.random.shuffle(nodes_left)

    while len(nodes_left) > 0:
        cluster_size = np.random.poisson(average_class_size)

        if cluster_size > len(nodes_left):
            break

        group = nodes_left[:cluster_size]
        groups.append(group)
        nodes_left = nodes_left[cluster_size:]

    for i in nodes_left:
        ng = np.random.choice(len(groups))
        groups[ng].append(i)

    if return_edges:
        for ng in range(len(groups)):
            group = groups[ng]
            Gn = nx.complete_graph(len(group))
            for e in Gn.edges():
                i, j = e
                node_i = group[i]
                node_j = group[j]
                G.add_edge(node_i, node_j)

    if verbose:
        if return_edges:
            ecount = np.zeros((len(age_keys), len(age_keys)))
            for e in G.edges():
                i, j = e

                age_i = age_by_uid_dic[i]
                index_i = age_keys_indices[age_i]
                age_j = age_by_uid_dic[j]
                index_j = age_keys_indices[age_j]

                ecount[index_i][index_j] += 1
                ecount[index_j][index_i] += 1

            print('within school age mixing matrix')
            print(ecount.astype(int))

    if return_edges:
        return list(G.edges())

    else:
        # if returning groups, much easier to add to population dictionaries and assign teachers to a single class
        return groups


def add_contacts_from_edgelist(popdict, edgelist, setting):

    for e in edgelist:
        i, j = e

        popdict[i]['contacts'][setting].add(j)
        popdict[j]['contacts'][setting].add(i)

    return popdict


def add_contacts_from_group(popdict, group, setting):

    for i in group:
        popdict[i]['contacts'][setting] = popdict[i]['contacts'][setting].union(group)
        popdict[i]['contacts'][setting].remove(i)

    return popdict


def add_contacts_from_groups(popdict, groups, setting):

    for group in groups:
        add_contacts_from_group(popdict, group, setting)

    return popdict


def generate_edges_between_teachers(teachers, average_teacher_teacher_degree):
    edges = []
    if average_teacher_teacher_degree > len(teachers):
        eiter = combinations(teachers, 2)
        edges = [e for e in eiter]

    else:
        p = average_teacher_teacher_degree/len(teachers)
        G = nx.erdos_renyi_graph(len(teachers), p)
        for e in G.edges():
            i, j = e
            teacher_i = teachers[i]
            teacher_j = teachers[j]
            e = (teacher_i, teacher_j)
            edges.append(e)
    return edges


def generate_edges_for_teachers_in_random_classes(syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, average_student_teacher_ratio=20, average_teacher_teacher_degree=4, verbose=False):

    age_keys = list(set(syn_school_ages))

    # create a dictionary with the list of uids for each age/grade
    uids_in_school_by_age = {}
    for a in age_keys:
        uids_in_school_by_age[a] = []

    for uid in syn_school_uids:
        a = age_by_uid_dic[uid]
        uids_in_school_by_age[a].append(uid)

    edges = []

    teachers_assigned = []
    available_teachers = deepcopy(teachers)
    for a in uids_in_school_by_age:

        n_teachers_needed = int(np.round(len(uids_in_school_by_age[a])/average_student_teacher_ratio, 1))
        n_teachers_needed = max(1, n_teachers_needed)  # at least one teacher

        if n_teachers_needed > len(available_teachers):
            selected_teachers = np.random.choice(teachers_assigned, replace=False, size=n_teachers_needed)
        else:
            selected_teachers = np.random.choice(available_teachers, replace=False, size=n_teachers_needed)
            for t in selected_teachers:
                available_teachers.remove(t)
                teachers_assigned.append(t)

        # only adds one teacher per student
        for student in uids_in_school_by_age[a]:
            teacher = np.random.choice(selected_teachers)
            e = (student, teacher)
            edges.append(e)

    # some teachers left so add them as contacts to other students
    for teacher in available_teachers:

        n_students = np.random.poisson(average_student_teacher_ratio)

        if n_students > len(syn_school_uids):
            n_students = len(syn_school_uids)

        selected_students = np.random.choice(syn_school_uids, replace=False, size=n_students)

        for student in selected_students:
            e = (student, teacher)
            edges.append(e)

        teachers_assigned.append(teacher)

    available_teachers = []

    teacher_teacher_edges = generate_edges_between_teachers(teachers_assigned, average_teacher_teacher_degree)
    edges += teacher_teacher_edges

    if verbose:
        G = nx.Graph()
        G.add_edges_from(edges)

        for s in syn_school_uids:
            print('student', s, 'contacts with teachers', G.degree(s))
        for t in teachers_assigned:
            print('teacher', t, 'contacts with students', G.degree(t))

    # not returning student-student contacts
    return edges


def generate_edges_for_teachers_in_clustered_classes(groups, teachers, average_student_teacher_ratio=20, average_teacher_teacher_degree=4, return_edges=False, verbose=False):

    edges = []
    teachers_assigned = []
    teacher_groups = []

    available_teachers = deepcopy(teachers)

    for ng, group in enumerate(groups):
        n_teachers_needed = int(np.round(len(group)/average_student_teacher_ratio, 1))
        n_teachers_needed = max(1, n_teachers_needed)

        if n_teachers_needed > len(available_teachers):
            selected_teachers = np.random.choice(teachers_assigned, replace=False, size=n_teachers_needed)
        else:
            selected_teachers = np.random.choice(available_teachers, replace=False, size=n_teachers_needed)
            for t in selected_teachers:
                available_teachers.remove(t)
                teachers_assigned.append(t)

        teacher_groups.append(list(selected_teachers))

        for student in group:
            teacher = np.random.choice(selected_teachers)
            e = (student, teacher)
            edges.append(e)

    # contacts are clustered so find a class to add to
    for teacher in available_teachers:
        ng = np.random.choice(np.arange(len(groups)))
        group = groups[ng]

        for student in group:
            e = (student, teacher)
            edges.append(e)

        teacher_groups[ng].append(teacher)

    if return_edges:
        teacher_teacher_edges = []
        for ng, teacher_group in enumerate(teacher_groups):
            teacher_teacher_edges += generate_edges_between_teachers(teacher_group, average_teacher_teacher_degree)
        edges += teacher_teacher_edges
        return edges
    else:
        return groups, teacher_groups


def add_school_edges(popdict, syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, average_student_teacher_ratio=20, average_teacher_teacher_degree=4, school_mixing_type='random', verbose=False):
    if school_mixing_type == 'random':
        edges = generate_random_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing, verbose)
        teacher_edges = generate_edges_for_teachers_in_random_classes(syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, average_student_teacher_ratio, average_teacher_teacher_degree, verbose)
        edges += teacher_edges
        add_contacts_from_edgelist(popdict, edges, 'S')

    elif school_mixing_type == 'clustered':

        student_groups = generate_clustered_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing, verbose=verbose)
        student_groups, teacher_groups = generate_edges_for_teachers_in_clustered_classes(student_groups, teachers, average_student_teacher_ratio, average_teacher_teacher_degree, verbose=verbose)

        for ng in range(len(student_groups)):
            student_group = student_groups[ng]
            teacher_group = teacher_groups[ng]
            group = student_group
            group += teacher_group
            add_contacts_from_group(popdict, group, 'S')

        # # additional edges between teachers in different classes - makes distint clusters connected
        # teacher_edges = generate_edges_between_teachers(teachers, average_teacher_teacher_degree-1)
        # add_contacts_from_edgelist(popdict, teacher_edges, 'S')

    # for i in teachers:
    #     con = popdict[i]['contacts']['S']
    #     print(i, len([c for c in con if c in teachers]))

    return popdict


grade_age_mapping = {i: i+5 for i in range(13)}
age_grade_mapping = {i+5: i for i in range(13)}

syn_school_ages = [5, 6, 8, 5, 9, 7, 8, 9, 5, 6, 7, 8, 8, 9, 9, 5, 6, 7, 8, 9, 5, 6, 8, 9, 9, 5, 6, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7,
                   6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 
                   9, 5, 8, 7, 8]


syn_school_uids = np.random.choice(np.arange(250), replace=False, size=len(syn_school_ages))

age_by_uid_dic = {}

for n in range(len(syn_school_uids)):
    uid = syn_school_uids[n]
    a = syn_school_ages[n]
    age_by_uid_dic[uid] = a

average_class_size = 20
inter_grade_mixing = 0.1
average_student_teacher_ratio = 20
average_teacher_teacher_degree = 4

teachers = list(np.random.choice(np.arange(250, 300), replace=False, size=int(math.ceil(len(syn_school_uids)/average_class_size))))

popdict = {}
for i in syn_school_uids:
    popdict.setdefault(i, {'contacts': {}})
    for k in ['H', 'S', 'W', 'C']:
        popdict[i]['contacts'][k] = set()

for i in teachers:
    popdict.setdefault(i, {'contacts': {}})
    for k in ['H', 'S', 'W', 'C']:
        popdict[i]['contacts'][k] = set()

# # test random
# # edges = generate_random_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing)
# # teacher_edges = generate_edges_for_teachers_in_random_classes(syn_school_uids, syn_school_ages, teachers, average_student_teacher_ratio=20, verbose=False)
add_school_edges(popdict, syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing, average_student_teacher_ratio, average_teacher_teacher_degree, school_mixing_type='random', verbose=False)


# test clustered
# groups = generate_clustered_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, return_edges=True, verbose=True)
# student_groups, teacher_groups = generate_edges_for_teachers_in_clustered_classes(popdict, groups, teachers, average_student_teacher_ratio=20, average_teacher_teacher_degree=4, return_edges=True, verbose=False)
# add_school_edges(popdict, syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing, average_student_teacher_ratio, average_teacher_teacher_degree, school_mixing_type='clustered', verbose=False)


ages_in_school_count = Counter(syn_school_ages)
# school_types_by_age = {}

# school_type_age_ranges = {}
# school_type_age_ranges['pk'] = np.arange(3, 6)
# school_type_age_ranges['es'] = np.arange(6, 11)
# school_type_age_ranges['ms'] = np.arange(11, 14)
# school_type_age_ranges['hs'] = np.arange(14, 18)
# school_type_age_ranges['uv'] = np.arange(18, 100)

# for a in range(100):
#     school_types_by_age[a] = dict.fromkeys(list(school_type_age_ranges.keys()), 0)

# for k in school_type_age_ranges.keys():
#     for a in school_type_age_ranges[k]:
#         school_types_by_age[a][k] = 1.

school_types_by_age = sp.get_default_school_types_by_age()

school_type_age_ranges = sp.get_default_school_type_age_ranges()

location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'

school_enrollment_counts_available = True
use_default = False


school_size_brackets = sp.get_default_school_size_distr_brackets()

school_size_distr_by_type = sp.get_default_school_size_distr_by_type()

# school_size_brackets = sp.get_school_size_brackets(sp.datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

# school_size_distr_by_type = {}
# school_size_distr_by_type['es'] = {i: 1./len(school_size_brackets) for i in school_size_brackets}
# school_size_distr_by_type['es'][1] += school_size_distr_by_type['es'][0]
# school_size_distr_by_type['es'][0] = 0

# for k in school_type_age_ranges:
#     school_size_distr_by_type[k] = school_size_distr_by_type['es']


uids_in_school = {syn_school_uids[n]: syn_school_ages[n] for n in range(len(syn_school_uids))}

# print(uids_in_school)

uids_in_school_by_age = {}
for a in range(100):
# for a in sorted(set(syn_school_ages)):
    uids_in_school_by_age[a] = []
for uid in uids_in_school:
    a = uids_in_school[uid]
    uids_in_school_by_age[a].append(uid)
ages_in_school_count = dict(Counter(syn_school_ages))
for a in range(100):
    if a not in ages_in_school_count:
        ages_in_school_count[a] = 0
ages_in_school_distr = sp.norm_dic(ages_in_school_count)


achoice = np.random.multinomial(1, [ages_in_school_distr[a] for a in ages_in_school_distr])
aindex = np.where(achoice)[0][0]


syn_schools, syn_school_uids = sp.send_students_to_school_with_school_types(school_size_distr_by_type, school_size_brackets, uids_in_school, uids_in_school_by_age,
                                                                            ages_in_school_count,
                                                                            school_types_by_age,
                                                                            school_type_age_ranges,
                                                                            verbose=False)


for ns in range(len(syn_schools)):
    print(ns, syn_schools[ns])






