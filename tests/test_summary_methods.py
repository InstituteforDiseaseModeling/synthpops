"""
Test summary methods for the generated population and compare to expected
demographic distributions where possible.
"""
import numpy as np
import sciris as sc
import synthpops as sp
import covasim as cv
import pytest
import settings


# parameters to generate a test population
pars = dict(
    n                       = settings.pop_sizes.small,
    rand_seed               = 123,

    smooth_ages             = True,

    with_facilities         = 1,
    with_non_teaching_staff = 1,
    with_school_types       = 1,

    school_mixing_type      = {'pk': 'age_and_class_clustered',
                               'es': 'age_and_class_clustered',
                               'ms': 'age_and_class_clustered',
                               'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
)
pars = sc.objdict(pars)


def test_summary_in_generation():
    """
    Test that summaries are produced produced when synthpops generates
    populations.
    """
    sp.logger.info("Test summaries are produced when populations are generated.")

    pop = sp.Pop(**pars)

    assert isinstance(pop.age_count, dict), 'Check failed'
    print(f"Age count summary exists and is a dictionary. The age range is from {min(pop.age_count.keys())} to {max(pop.age_count.keys())} years old.")


def test_change_sheet_name():
    """
    Test that parameters can be changed from defaults.
    """
    sp.logger.info("Test that parameters can be changed from defaults and used.")
    test_pars = sc.dcp(pars)
    test_pars.sheet_name = 'Senegal'
    pop = sp.Pop(**test_pars)

    expected_contact_matrix_dic = sp.get_contact_matrix_dic(sp.datadir, sheet_name=test_pars.sheet_name)
    r, c = expected_contact_matrix_dic['H'].shape

    # check that the correct contact matrices are used in population generation
    for k in expected_contact_matrix_dic.keys():
        print(pop.contact_matrix_dic[k][0, 0], expected_contact_matrix_dic[k][0, 0])
        print(type(pop.contact_matrix_dic[k]), type(expected_contact_matrix_dic[k]))

        for ri in range(r):
            for ci in range(c):
                np.testing.assert_array_equal([pop.contact_matrix_dic[k][ri, ci]], [expected_contact_matrix_dic[k][ri, ci]], err_msg=f"{ri, ci, pop.contact_matrix_dic[k][ri, ci], expected_contact_matrix_dic[k][ri, ci], k}")

        # np.testing.assert_allclose(pop.contact_matrix_dic[k], expected_contact_matrix_dic[k])
        # assert pop.contact_matrix_dic[k].all() == expected_contact_matrix_dic[k].all(), f"Check failed, contact matrices used in population generation don't match the ones expected from {test_pars.sheet_name}."
    print(f"Check passed. {test_pars.sheet_name} contact matrices used in population generation.")


if __name__ == '__main__':

    # test_summary_in_generation()
    test_change_sheet_name()
