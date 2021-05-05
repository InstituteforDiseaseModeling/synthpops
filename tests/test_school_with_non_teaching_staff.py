"""
Test that the parameter with_non_teaching_staff is being used to decide whether
to include non teaching staff in schools.

"""

import os
import sciris as sc
import synthpops as sp
import numpy as np
import settings


# parameters to generate a test population
pars = sc.objdict(
        n                               = settings.pop_sizes.small,
        rand_seed                       = 123,
        with_non_teaching_staff         = 1,

)


def test_with_non_teaching_staff():
    """
    When with_non_teaching_staff is True, each school should be created with non
    teaching staff. Otherwise when False, there should be no non teaching staff
    in schools.

    """
    sp.logger.info(f'Testing the effect of the parameter with_non_teaching_staff.')

    test_pars = sc.dcp(pars)
    test_pars.with_non_teaching_staff = True
    pop_1 = sp.Pop(**test_pars)
    popdict_1 = pop_1.to_dict()

    test_pars.with_non_teaching_staff = False
    pop_2 = sp.Pop(**test_pars)
    popdict_2 = pop_2.to_dict()

    school_staff_1 = {}
    for i, person in popdict_1.items():
        if person['scid'] is not None:
            school_staff_1.setdefault(person['scid'], 0)
        if person['sc_staff']:
            school_staff_1[person['scid']] += 1

    staff_ids_2 = set()
    for i, person in popdict_2.items():
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
