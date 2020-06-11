"""
This module generates school contacts by class and grade in flexible ways.
Contacts can be clustered into classes and also mixed across the grade and
across the school.

H. Guclu et. al (2016) shows that mixing across grades is low for public
schools in elementary and middle schools. Mixing across grades is however
higher in high schools.

Functions in this module are flexible to allow users to specify the
inter-grade mixing, and to choose whether contacts are clustered within a
grade. Clustering contacts across different grades is not supported because
there is no data to suggest that this happens commonly.

"""

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

from . import base as spb
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contacts as spct
from .config import datadir


# adding edges to the popdict, either from an edgelist or groups (groups are better when you have fully connected graphs - no need to enumerate for n*(n-1)/2 edges!)
def add_contacts_from_edgelist(popdict, edgelist, setting):
    """
    Adds contacts to popdict from edges in an edgelist.

    Args:
        popdict (dict)  : dict of people
        edgelist (list) : list of edges
        setting (str)   : social setting layer

    Returns:
        Updated popdict.
    """

    for e in edgelist:
        i, j = e

        popdict[i]['contacts'][setting].add(j)
        popdict[j]['contacts'][setting].add(i)

    return popdict


def add_contacts_from_group(popdict, group, setting):
    """
    Adds contacts to popdict from fully connected group.

    Args:
        popdict (dict) : dict of people
        group (list)   : list of people in group
        setting (str)  : social setting layer

    Returns:
        Updated popdict.
    """

    for i in group:
        popdict[i]['contacts'][setting] = popdict[i]['contacts'][setting].union(group)
        popdict[i]['contacts'][setting].remove(i)

    return popdict


def add_contacts_from_groups(popdict, groups, setting):
    """
    Adds contacts to popdict from fully connected groups.

    Args:
        popdict (dict) : dict of people
        groups (list)  : list of lists of people in groups
        setting (str)  : social setting layer

    Returns:
        Updated popdict.
    """
    for group in groups:
        add_contacts_from_group(popdict, group, setting)

    return popdict


def generate_random_contacts_across_school(all_school_uids, average_class_size, verbose=False):
    edges = []
    p = average_class_size/len(all_school_uids)
    G = nx.erdos_renyi_graph(len(all_school_uids), p)
    for e in G.edges():
        i, j = e
        node_i = all_school_uids[i]
        node_j = all_school_uids[j]
        e = (node_i, node_j)
        edges.append(e)
    return edges


def generate_random_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, verbose=False):
    """
    Generates edges for contacts mostly within the same age/grade. Edges are randomly distributed so that clustering is roughly average_class_size/size of the grade.
    Inter grade mixing is done by rewiring edges, specifically swapping endpoints of pairs of randomly sampled edges.

    Args:
        syn_school_uids (list)     : list of uids of students in the school
        syn_school_ages (list)     : list of the ages of the students in the school
        age_by_uid_dic (dict)      : dict mapping uid to age
        grade_age_mapping (dict)   : dict mapping grade to an age
        age_grade_mapping (dict)   : dict mapping age to a grade
        average_class_size (int)   : average class size
        inter_grade_mixing (float) : percent of within grade edges that rewired to create edges across grades
        verbose (bool)             : print statements throughout

    Returns:
        List of edges between students in school.

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
    E = list(G.edges())
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
    # verbose = True
    # print(syn_school_uids, 'ids')
    # print(syn_school_ages, 'ages')
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
    Generates edges for contacts mostly within the same age/grade. Edges are randomly distributed so that clustering is roughly average_class_size/size of the grade.
    Inter grade mixing is done by rewiring edges, specifically swapping endpoints of pairs of randomly sampled edges.

    Args:
        syn_school_uids (list)     : list of uids of students in the school
        syn_school_ages (list)     : list of the ages of the students in the school
        age_by_uid_dic (dict)      : dict mapping uid to age
        grade_age_mapping (dict)   : dict mapping grade to an age
        age_grade_mapping (dict)   : dict mapping age to a grade
        average_class_size (int)   : average class size
        inter_grade_mixing (float) : percent of within grade edges that rewired to create edges across grades
        return_edges (bool)        : If True, return edges, else return two groups of contacts - students and teachers for each class
        verbose (bool)             : print statements throughout

    Returns:
        List of edges between students in school or groups of contacts.

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

    if len(groups) == 0:
        groups.append(nodes_left)
        nodes_left = []

    while len(nodes_left) > 0:
        cluster_size = np.random.poisson(average_class_size)

        if cluster_size > len(nodes_left):
            break

        group = nodes_left[:cluster_size]
        groups.append(group)
        nodes_left = nodes_left[cluster_size:]

    for i in nodes_left:
        ng = np.random.choice(a=np.arange(len(groups)))  # choose one of the other classes to add to
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


def generate_edges_between_teachers(teachers, average_teacher_teacher_degree):
    """
    Generate edges between teachers.

    Args:
        teachers (list): a list of teachers
        average_teacher_teacher_degree (int): average number of contacts with other teachers

    Return:
        List of edges between teachers.

    """
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
    """
    Generate edges for teachers, including to both students and other teachers at the same school.
    Well mixed contacts within the same age/grade, some cross grade mixing. Teachers are clustered by grade mostly.

    Args:
        syn_school_uids (list)               : list of uids of students in the school
        syn_school_ages (list)               : list of the ages of the students in the school
        teachers (list)                      : list of teachers in the school
        age_by_uid_dic (dict)                : dict mapping uid to age
        grade_age_mapping (dict)             : dict mapping grade to an age
        age_grade_mapping (dict)             : dict mapping age to a grade
        average_student_teacher_ratio (int)  : average number of students per teacher
        average_teacher_teacher_degree (int) : average number of contacts with other teachers
        verbose (bool)                       : print statements throughout

    Return:
        List of edges connected to teachers.

    """
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

        if n_teachers_needed > len(available_teachers) + len(teachers_assigned):
            n_teachers_needed = len(available_teachers) + len(teachers_assigned)
            selected_teachers = available_teachers + teachers_assigned

        elif n_teachers_needed > len(available_teachers):
            selected_teachers = available_teachers
            n_teachers_needed = n_teachers_needed - len(available_teachers)
            selected_teachers += list(np.random.choice(teachers_assigned, replace=False, size=n_teachers_needed))

            # selected_teachers = np.random.choice(teachers_assigned, replace=False, size=n_teachers_needed)
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
    """
    Generate edges for teachers, including to both students and other teachers at the same school.
    Students and teachers are clustered into disjoint classes.

    Args:
        groups (list)                        : list of lists of students, clustered into groups mostly by grade
        teachers (list)                      : list of teachers in the school
        average_student_teacher_ratio (int)  : average number of students per teacher
        average_teacher_teacher_degree (int) : average number of contacts with other teachers
        return_edges (bool)                  : If True, return edges, else return two groups of contacts - students and teachers for each class
        verbose (bool)                       : print statements throughout

    Return:
        List of edges connected to teachers.

    """
    edges = []
    teachers_assigned = []
    teacher_groups = []

    np.random.shuffle(groups)  # shuffle the clustered groups of students / classes so that the classes aren't ordered from youngest to oldest

    available_teachers = deepcopy(teachers)

    # have exactly as many teachers as needed
    if len(groups) == len(available_teachers):
        for ng, t in enumerate(available_teachers):
            teacher_groups.append(list(t))
        teachers_assigned = teachers
        available_teachers = []

    # you don't have enough teachers to cover the classes so break the extra groups up
    elif len(groups) > len(available_teachers):
        n_groups_to_break = len(groups) - len(available_teachers)

        # grab the last cluster and split it up and spread the students to the other groups
        for ngb in range(n_groups_to_break):
            group_to_break = groups[-1]

            for student in group_to_break:
                ng = np.random.randint(len(groups) - 1)  # find another class to join
                groups[ng].append(student)
            groups = groups[:-1]

        for ng, t in enumerate(available_teachers):
            teacher_groups.append(list(t))
        teachers_assigned = teachers
        available_teachers = []

    elif len(groups) < len(available_teachers):
        for ng, group in enumerate(groups):

            # class size already determines that each class gets at least one teacher - maybe we can add other teachers some other way
            teacher_groups.append(list(available_teachers[ng]))
        available_teachers = available_teachers[len(groups):]

        # spread extra teachers among the classes
        for t in available_teachers:
            ng = np.random.randint(len(groups))
            teacher_groups[ng].append(t)
        teachers_assigned = teachers
        available_teachers = []

    # print('available_teachers', len(available_teachers), [len(g) for g in groups])

    # for ng, group in enumerate(groups):
    #     n_teachers_needed = int(np.round(len(group)/average_student_teacher_ratio, 1))
    #     n_teachers_needed = max(1, n_teachers_needed)
    #     # print('nt', n_teachers_needed, len(available_teachers), len(teachers_assigned))

    #     if n_teachers_needed > len(available_teachers) + len(teachers_assigned):
    #         n_teachers_needed = len(available_teachers) + len(teachers_assigned)

    #         selected_teachers = available_teachers + teachers_assigned

    #     elif n_teachers_needed > len(available_teachers):
    #         selected_teachers = available_teachers
    #         n_teachers_needed = n_teachers_needed - len(available_teachers)
    #         selected_teachers += list(np.random.choice(teachers_assigned, replace=False, size=n_teachers_needed))

    #     else:
    #         selected_teachers = np.random.choice(available_teachers, replace=False, size=n_teachers_needed)
    #         for t in selected_teachers:
    #             available_teachers.remove(t)
    #             teachers_assigned.append(t)

    #     teacher_groups.append(list(selected_teachers))

    #     for student in group:
    #         teacher = np.random.choice(selected_teachers)
    #         e = (student, teacher)
    #         edges.append(e)

    # contacts are clustered so find a class to add to
    # for teacher in available_teachers:
        # ng = np.random.choice(np.arange(len(groups)))
        # group = groups[ng]

    for ng, group in enumerate(groups):
        for student in group:
            for teacher in teacher_groups[ng]:
                e = (student, teacher)
                edges.append(e)

        # teacher_groups[ng].append(teacher)

    if return_edges:
        teacher_teacher_edges = []
        for ng, teacher_group in enumerate(teacher_groups):
            teacher_teacher_edges += generate_edges_between_teachers(teacher_group, average_teacher_teacher_degree)
        edges += teacher_teacher_edges
        # not returning student-student contacts
        return edges
    else:
        return groups, teacher_groups


def add_school_edges(popdict, syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, average_student_teacher_ratio=20, average_teacher_teacher_degree=4, school_mixing_type='random', verbose=False):
    """
    Generate edges for teachers, including to both students and other teachers at the same school.

    Args:
        popdict (dict)                       : dictionary of people
        syn_school_uids (list)               : list of uids of students in the school
        syn_school_ages (list)               : list of the ages of the students in the school
        teachers (list)                      : list of teachers in the school
        age_by_uid_dic (dict)                : dict mapping uid to age
        grade_age_mapping (dict)             : dict mapping grade to an age
        age_grade_mapping (dict)             : dict mapping age to a grade
        average_class_size (int)             : average class size
        average_student_teacher_ratio (int)  : average number of students per teacher
        average_teacher_teacher_degree (int) : average number of contacts with other teachers
        school_mixing_type(str)              : 'random' for well mixed schools, 'clustered' for disjoint classes in a school
        verbose (bool)                       : print statements throughout

    Return:
        Updated popdict.

    """

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


def get_default_school_type_age_ranges():

    school_type_age_ranges = {}
    school_type_age_ranges['pk'] = np.arange(3, 6)
    school_type_age_ranges['es'] = np.arange(6, 11)
    school_type_age_ranges['ms'] = np.arange(11, 14)
    school_type_age_ranges['hs'] = np.arange(14, 18)
    school_type_age_ranges['uv'] = np.arange(18, 100)

    return school_type_age_ranges


def get_default_school_types_by_age():

    school_type_age_ranges = get_default_school_type_age_ranges()

    school_types_by_age = {}
    for a in range(100):
        school_types_by_age[a] = dict.fromkeys(list(school_type_age_ranges.keys()), 0.)

    for k in school_type_age_ranges.keys():
        for a in school_type_age_ranges[k]:
            school_types_by_age[a][k] = 1.

    return school_types_by_age


def get_default_school_size_distr_brackets():
    return spdata.get_school_size_brackets(datadir, country_location='default', use_default=True)


def get_default_school_size_distr_by_type():
    school_size_distr_by_type = {}

    school_types = ['pk', 'es', 'ms', 'hs', 'uv']

    for k in school_types:
        school_size_distr_by_type[k] = spdata.get_school_size_distr_by_brackets(datadir, country_location='default', use_default=True)

    return school_size_distr_by_type

