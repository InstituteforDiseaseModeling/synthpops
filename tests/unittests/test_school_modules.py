'''
Test creation of schools
'''

from collections import Counter
from itertools import combinations
import sciris as sc
import numpy as np
import networkx as nx
import synthpops as sp
from synthpops import schools as spsch


def add_contacts_from_groups(popdict, groups, setting):
    """
    Add contacts to popdict from fully connected groups. Note that this simply adds to the contacts already in the layer and does not overwrite the contacts.

    Args:
        popdict (dict) : dict of people
        groups (list)  : list of lists of people in group
        setting (str)  : social setting layer

    Returns:
        Updated popdict.

    """
    for group in groups:
        spsch.add_contacts_from_group(popdict, group, setting)

    return popdict


# DM: note to self, the 'test' below is not an actual test and should be cleaned up/modified
def test_school_modules():

    grade_age_mapping = {i: i + 5 for i in range(13)}
    age_grade_mapping = {i + 5: i for i in range(13)}

    syn_school_ages = [5, 6, 8, 5, 9, 7, 8, 9, 5, 6, 7, 8, 8, 9, 9, 5, 6, 7, 8, 9, 5, 6, 8, 9, 9, 5, 6, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7,
                       6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 5, 7, 5, 7, 7, 8, 6, 5, 6, 7, 8, 9, 5, 6, 6, 7, 8, 9, 9, 5, 6, 7, 7, 8, 9, 6, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7, 6, 7, 7, 7, 5, 6, 8, 8, 9, 9, 5, 8, 9, 6, 5, 7, 9, 7, 8, 9, 5, 6, 8, 8, 6, 5, 7,
                       9, 5, 8, 7, 8]

    syn_school_uids = list(np.random.choice(np.arange(250), replace=False, size=len(syn_school_ages)))

    age_by_uid_dic = {}

    for n in range(len(syn_school_uids)):
        uid = syn_school_uids[n]
        a = syn_school_ages[n]
        age_by_uid_dic[uid] = a

    average_class_size = 20
    inter_grade_mixing = 0.1
    average_student_teacher_ratio = 20
    average_teacher_teacher_degree = 4
    average_additional_staff_degree = 15

    teachers = list(np.random.choice(np.arange(250, 300), replace=False, size=int(np.ceil(len(syn_school_uids) / average_class_size))))

    popdict = {}
    for i in syn_school_uids:
        popdict.setdefault(i, {'contacts': {}})
        for k in ['H', 'S', 'W', 'C']:
            popdict[i]['contacts'][k] = set()

    for i in teachers:
        popdict.setdefault(i, {'contacts': {}})
        for k in ['H', 'S', 'W', 'C']:
            popdict[i]['contacts'][k] = set()

    # empty non teaching staff
    non_teaching_staff = []

    # # test random
    # # edges = generate_random_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing)
    # # teacher_edges = generate_edges_for_teachers_in_random_classes(syn_school_uids, syn_school_ages, teachers, average_student_teacher_ratio=20)
    spsch.add_school_edges(popdict,
                           syn_school_uids,
                           syn_school_ages,
                           teachers,
                           non_teaching_staff,
                           age_by_uid_dic,
                           grade_age_mapping,
                           age_grade_mapping,
                           average_class_size,
                           inter_grade_mixing,
                           average_student_teacher_ratio,
                           average_teacher_teacher_degree,
                           average_additional_staff_degree,
                           school_mixing_type='random')

    # test clustered
    # groups = generate_clustered_classes_by_grade_in_school(syn_school_uids, syn_school_ages, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, return_edges=True)
    # student_groups, teacher_groups = generate_edges_for_teachers_in_clustered_classes(popdict, groups, teachers, average_student_teacher_ratio=20, average_teacher_teacher_degree=4, return_edges=True)
    # add_school_edges(popdict, syn_school_uids, syn_school_ages, teachers, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing, average_student_teacher_ratio, average_teacher_teacher_degree, school_mixing_type='clustered')

    ages_in_school_count = Counter(syn_school_ages)
    school_types_distr_by_age = sp.get_default_school_types_distr_by_age()
    school_type_age_ranges = sp.get_default_school_type_age_ranges()
    school_size_brackets = sp.get_default_school_size_distr_brackets()
    school_size_distr_by_type = sp.get_default_school_size_distr_by_type()
    uids_in_school = {syn_school_uids[n]: syn_school_ages[n] for n in range(len(syn_school_uids))}

    uids_in_school_by_age = {}
    for a in range(100):
        uids_in_school_by_age[a] = []
    for uid in uids_in_school:
        a = uids_in_school[uid]
        uids_in_school_by_age[a].append(uid)
    ages_in_school_count = dict(Counter(syn_school_ages))
    for a in range(100):
        if a not in ages_in_school_count:
            ages_in_school_count[a] = 0

    syn_schools, syn_school_uids, syn_school_types = spsch.send_students_to_school_with_school_types(school_size_distr_by_type,
                                                                                                      school_size_brackets,
                                                                                                      uids_in_school,
                                                                                                      uids_in_school_by_age,
                                                                                                      ages_in_school_count,
                                                                                                      school_types_distr_by_age,
                                                                                                      school_type_age_ranges,
                                                                                                      )

    for ns in range(len(syn_schools)):
        print(ns, syn_schools[ns])

    return syn_schools, syn_school_uids, syn_school_types


def test_debug_log_for_school_methods():
  """Test that setting logger level to DEBUG prints statements for different school mixing methods"""
  sp.logger.setLevel('DEBUG')
  
  # without school mixing types defined
  pars = sc.objdict(n=1e3, with_school_types=0, school_mixing_type='random')
  pop = sp.Pop(**pars)

  # school types defined, mixing type set to random
  pars.with_school_types = 1
  pars.school_mixing_type = 'random'
  pop = sp.Pop(**pars)

  # school types defined, mixing type set to age_clustered
  pars.school_mixing_type = 'age_clustered'
  pop = sp.Pop(**pars)

  # school types defined, mixing type set to age_and_class_clustered
  pars.school_mixing_type = 'age_and_class_clustered'
  pop = sp.Pop(**pars)
  sp.logger.setLevel('INFO')  # need to reset logger level - this changes a synthpops setting


if __name__ == '__main__':
    syn_schools, syn_school_uids, syn_school_types = test_school_modules()
    test_debug_log_for_school_methods()