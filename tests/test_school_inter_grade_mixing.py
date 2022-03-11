"""
Test that the inter_grade_mixing parameter is being used for specific school
mixing types.

We expect inter_grade_mixing to be used when school_mixing_type equals
'age_clustered', and to be ignored when school_mixing_type is equal to either
'random' or 'age_and_class_clustered'.
"""

import os
import sciris as sc
import synthpops as sp
import numpy as np
import settings

# parameters to generate a test population
pars = sc.objdict(
        n                       = settings.pop_sizes.medium,
        rand_seed               = 23,
        with_facilities         = 1,
        with_non_teaching_staff = 1,
        with_school_types       = 1,

        school_mixing_type      = 'age_clustered',
        inter_grade_mixing      = 0.1,
        average_class_size      = 30,

)


def test_inter_grade_mixing(school_mixing_type='random'):
    """
    When school_mixing_type is 'age_clustered' then inter_grade_mixing should
    rewire a fraction of the edges between students in the same age or grade to
    be edges with any other student in the school.

    When school_mixing_type is 'random' or 'age_and_class_clustered',
    inter_grade_mixing has no effect.

    Args:
        school_mixing_type (str): The mixing type for schools, 'random', 'age_clustered', or 'age_and_class_clustered'.

    """
    sp.logger.info(f'Testing the effect of the parameter inter_grade_mixing for school_mixing_type: {school_mixing_type}')

    test_pars = sc.dcp(pars)
    test_pars.school_mixing_type = school_mixing_type

    pop_1 = sp.Pop(**test_pars)
    popdict_1 = pop_1.to_dict()

    test_pars.inter_grade_mixing = 0.3
    pop_2 = sp.Pop(**test_pars)
    popdict_2 = pop_2.to_dict()

    # make an adjacency matrix of edges between students
    adjm_1 = np.zeros((101, 101))
    adjm_2 = np.zeros((101, 101))

    student_ids = set()
    for i, person in popdict_1.items():
        if person['sc_student']:
            student_ids.add(i)

    for ni, i in enumerate(student_ids):

        contacts_1 = popdict_1[i]['contacts']['S']
        contacts_2 = popdict_2[i]['contacts']['S']

        student_contacts_1 = list(set(contacts_1).intersection(student_ids))
        student_contacts_2 = list(set(contacts_2).intersection(student_ids))

        age_i = popdict_1[i]['age']

        for nj, j in enumerate(student_contacts_1):
            age_j = popdict_1[j]['age']
            adjm_1[age_i][age_j] += 1

        for nj, j in enumerate(student_contacts_2):
            age_j = popdict_2[j]['age']
            adjm_2[age_i][age_j] += 1

    if school_mixing_type in ['random', 'age_and_class_clustered']:
        assert np.not_equal(adjm_1, adjm_2).sum() == 0, f"inter_grade_mixing parameter check failed. Different values for this parameter produced different results for school_mixing_type: {school_mixing_type}."

    elif school_mixing_type in ['age_clustered']:
        assert np.not_equal(adjm_1, adjm_2).sum() > 0, f"inter_grade_mixing parameter check failed. It produced the same results for different values of this parameter for school_mixing_type: {school_mixing_type}."


if __name__ == '__main__':

    sc.tic()
    test_inter_grade_mixing('random')
    test_inter_grade_mixing('age_clustered')
    test_inter_grade_mixing('age_and_class_clustered')

    sc.toc()
