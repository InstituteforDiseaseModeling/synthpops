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
    sp.logger.info("Temporary basic tests.")

    pop = sp.Pop(**pars)

    assert isinstance(pop.age_count, dict), "Check failed"
    print(f"Age count summary exists and is a dictionary. The age range is from {min(pop.age_count.keys())} to {max(pop.age_count.keys())} years old.")

    assert sum(pop.enrollment_by_age.values()) > 0, "Check failed. Student enrollment is less than or equal to 0."
    print("Student enrollment count by age exists and is greater than 0.")

    enrollment_rates = pop.enrollment_rates  # a property rather than stored data
    assert 0 < enrollment_rates[10] <= 1., "Check failed. Enrollment rate for age 10 is less than or equal to 0."
    print(f"Enrollment rate for age 10 is {enrollment_rates[10] * 100:.2f}%.")

    employment_rates = pop.employment_rates
    assert 0 < employment_rates[25] <= 1., "Check failed. Employment rate for age 25 is less than or equal to 0."
    print(f"Employment rate for age 25 is {employment_rates[25] * 100:.2f}%.")

    workplace_sizes = pop.workplace_sizes
    assert sum(workplace_sizes.values()) > 0, "Check failed. Sum of workplace sizes is less than or equal to 0."
    print("Workplace sizes exists in pop object and is a dictionary by workplace id (wpid).")

    workplace_size_brackets = sp.get_workplace_size_brackets(**pop.loc_pars)
    workplace_size_bins = sp.get_bin_edges(workplace_size_brackets)
    workplace_size_bin_labels = sp.get_bin_labels(workplace_size_brackets)
    # print(workplace_size_bins)
    # print(workplace_size_bin_labels)

    workplace_size_dist = sp.get_generated_workplace_size_distribution(workplace_sizes, workplace_size_bins)
    # print(workplace_size_dist)
    expected_workplace_size_dist = sp.norm_dic(sp.get_workplace_size_distr_by_brackets(sp.datadir, state_location=pop.state_location, country_location=pop.country_location))
    # print(expected_workplace_size_dist)

    household_sizes = pop.household_sizes
    household_size_count = pop.household_size_count

    assert sum(household_size_count.values()) > 0, "Check failed. No people placed into unique households."
    print("Household sizes exist in pop object and is a dictionary by household id (hhid).")

    assert sum(household_sizes.values()) == sum([household_size_count[k] * k for k in household_size_count]), f"Household sizes summary check failed."
    print("Household sizes created.")

    ltcf_sizes = pop.ltcf_sizes
    ltcf_size_count = pop.ltcf_size_count

    assert sum(ltcf_sizes.values()) > 0, "Check failed. No people placed in ltcfs."
    print("Ltcfs created.")

    ltcf_sizes_res = pop.get_ltcf_sizes(keys_to_exclude=['snf_staff'])
    assert sum(ltcf_sizes_res.values()) < sum(ltcf_sizes.values()), "Check failed. Ltcf residents is greater than or equal to all people in ltcfs."
    print("Ltcf residents created separately.")

    assert sum(household_sizes.values()) + sum(ltcf_sizes_res.values()) == pop.n, f"Check failed. Population size is {pop.n} and the sum of people generated living in households and ltcfs is {sum(household_sizes.values()) + sum(ltcf_sizes_res.values())}."
    print("Check passed. Everyone lives either in a household or ltcf.")

    ltcf_sizes_staff = pop.get_ltcf_sizes(keys_to_exclude=['snf_res'])
    assert sum(ltcf_sizes_res.values()) + sum(ltcf_sizes_staff.values()) == sum(ltcf_sizes.values()), "Check failed. The sum of ltcf residets and staff counted separately does not equal the count of them together."
    print("Ltcf staff created separately.")

    head_ages = pop.get_household_head_ages()
    assert min(head_ages.values()) >= 18, "Check failed. Min head age is younger than 18 years old."
    print("Check passed. All heads of households are at least 18 years old.")


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
