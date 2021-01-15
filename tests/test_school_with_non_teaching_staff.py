"""
Test that the parameter with_non_teaching_staff is being used to decide whether
to include non teaching staff in schools.

"""

import os
import sciris as sc
import synthpops as sp
import numpy as np

# parameters to generate a test population
pars = dict(
        n                               = 20001,
        rand_seed                       = 123,
        max_contacts                    = None,

        with_industry_code              = 0,
        with_facilities                 = 1,
        with_non_teaching_staff         = 1,
        use_two_group_reduction         = 1,
        with_school_types               = 1,

        average_LTCF_degree             = 20,
        ltcf_staff_age_min              = 20,
        ltcf_staff_age_max              = 60,

        school_mixing_type              = 'age_clustered',
        average_class_size              = 20,
        inter_grade_mixing              = 0.1,
        teacher_age_min                 = 25,
        teacher_age_max                 = 75,
        staff_age_min                   = 20,
        staff_age_max                   = 75,

        average_student_teacher_ratio   = 20,
        average_teacher_teacher_degree  = 3,
        average_student_all_staff_ratio = 15,
        average_additional_staff_degree = 20,
)


def test_with_non_teaching_staff():
    """
    When with_non_teaching_staff is True, each school should be created with non
    teaching staff. Otherwise when False, there should be no non teaching staff
    in schools.

    """
    sp.logger.info(f'Testing the effect of the parameter with_non_teaching_staff.')

    test_pars = sc.dcp(pars)
    test_pars['with_non_teaching_staff'] = True

    pop_1 = sp.make_population(**test_pars)
    test_pars['with_non_teaching_staff'] = False
    pop_2 = sp.make_population(**test_pars)

    school_staff_1 = {}
    for i, person in pop_1.items():
        if person['scid'] is not None:
            school_staff_1.setdefault(person['scid'], 0)
        if person['sc_staff']:
            school_staff_1[person['scid']] += 1

    staff_ids_2 = set()
    for i, person in pop_2.items():
        if person['sc_staff']:
            staff_ids_2.add(i)

    # with_non_teaching_staff == True so minimum number of staff per school needs to be 1
    assert min(school_staff_1.values()) >= 1, f"with_non_teaching_staff parameter check failed when set to True."

    # with_non_teaching_staff == False so there should be no one who shows up with sc_staff set to 1
    assert len(staff_ids_2) == 0, f"with_non_teaching_staff parameter check failed when set to False."


if __name__ == '__main__':

    sc.tic()
    test_with_non_teaching_staff()
    sc.toc()
