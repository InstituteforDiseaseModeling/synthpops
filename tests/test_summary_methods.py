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
    n                       = settings.pop_sizes.medium,
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

    assert isinstance(pop.age_count, dict), "Check failed"
    print(f"Age count summary exists and is a dictionary. The age range is from {min(pop.age_count.keys())} to {max(pop.age_count.keys())} years old.")

    assert sum(pop.enrollment_by_age.values()) > 0, "Check failed. Student enrollment is not greater than 0."
    print("Student enrollment count by age exists and is greater than 0.")

    enrollment_rates = pop.enrollment_rates  # a property rather than stored data
    assert 0 < enrollment_rates[10] <= 1., "Check failed. Enrollment rate for age 10 is not greater than 0."
    print(f"Enrollment rate for age 10 is {enrollment_rates[10] * 100:.2f}%.")

    # print(pop.employment_by_age)
    employment_rates = pop.employment_rates
    assert 0 < employment_rates[25] <= 1., "Check failed. Employment rate for age 25 is not greater than 0."
    print(f"Employment rate for age 25 is {employment_rates[25] * 100:.2f}%.")

    # for i, person in pop.popdict.items():
    #     if i % 2 == 0:
    #         # if person['snf_res']:
    #             # print(i, person['age'], 'snf_res')
    #         if person['snf_staff']:
    #             print(i, person['age'], 'snf_staff')
    #         elif person['sc_teacher']:
    #             print(i, person['age'], 'sc_teacher')
    #         elif person['sc_staff']:
    #             print(i, person['age'], 'sc_staff')
    #         elif person['wpid']:
    #             print(i, person['age'], 'wpid')

    #     if i > 500:
    #         break


def test_contact_matrices_used():
    """
    Test that the contact matrices used in generation are left unmodified. The
    workplaces module instead should modify a copy of the workplace contact
    matrix and leave the original contact matrix as is. This means the matrices
    added to the pop object as a summary should match expected data.
    """
    sp.logger.info("Test that the contact matrices used in generation match the expected data.")
    pop = sp.Pop(**pars)

    expected_contact_matrix_dic = sp.get_contact_matrix_dic(sp.datadir, sheet_name=pop.sheet_name)
    for k in expected_contact_matrix_dic.keys():
        err_msg = f"Check failed for contact setting {k}."
        np.testing.assert_array_equal(pop.contact_matrix_dic[k], expected_contact_matrix_dic[k], err_msg=err_msg)

    print("Contact matrices check passed.")


def test_change_sheet_name():
    """
    Test that the sheet_name parameter can be changed from defaults.
    """
    sp.logger.info("Test that parameters can be changed from defaults and used.")
    test_pars = sc.dcp(pars)
    test_pars.sheet_name = 'Senegal'
    pop = sp.Pop(**test_pars)

    expected_contact_matrix_dic = sp.get_contact_matrix_dic(sp.datadir, sheet_name=test_pars.sheet_name)

    # check that the correct contact matrices are used in population generation
    for k in expected_contact_matrix_dic.keys():
        err_msg = f"Check failed for contact setting {k}."
        np.testing.assert_array_equal(pop.contact_matrix_dic[k], expected_contact_matrix_dic[k], err_msg=err_msg)

    print(f"Check passed. {test_pars.sheet_name} contact matrices used in population generation.")


if __name__ == '__main__':

    test_summary_in_generation()
    test_contact_matrices_used()
    test_change_sheet_name()
