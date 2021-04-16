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
pars = sc.objdict(
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


def test_summary_in_generation():
    """
    Basic tests that summaries are produced produced when synthpops generates
    populations. Summaries are stored and accessed via sp.Pop().summary.
    """
    sp.logger.info("Test summaries are produced when populations are generated.")
    sp.logger.info("Temporary basic tests. To be reorganized and converted to plotting based tests.\n")

    pop = sp.Pop(**pars)

    # check age_count summary
    assert isinstance(pop.summary.age_count, dict), "Check failed"
    print(f"Check passed. Age count summary exists and is a dictionary. The age range is from {min(pop.summary.age_count.keys())} to {max(pop.summary.age_count.keys())} years old.")

    assert sum(pop.summary.age_count.values()) == pop.n, f"Check failed. The sum of pop.age_count ({sum(pop.summary.age_count.values())}) does not equal the population size ({pop.n})."
    print(f"Check passed. Age count summary of the generated population matches the expected size ({pop.n}).\n")

    # check household size summary
    assert sum(pop.summary.household_size_count.values()) > 0, "Check failed. No people placed into unique households."
    print("Check passed. Household sizes exist in pop object and is a dictionary by household id (hhid).")

    assert sum(pop.summary.household_sizes.values()) == sum([pop.summary.household_size_count[k] * k for k in pop.summary.household_size_count]), "Household sizes summary check failed."
    print("Check passed. Household sizes summary check passed.\n")

    # check household size distribution
    household_size_dist = sp.get_generated_household_size_distribution(pop.summary.household_sizes)
    expected_household_size_dist = sp.get_household_size_distr(**pop.loc_pars)
    if expected_household_size_dist[1] > 0:
        assert household_size_dist[1] > 0, "Check failed. No one lives alone even though the expected household size distribution says otherwise."
        print("Check passed. At least some people live alone as expected from the household size distribution.")

    # check household head summary
    assert min(pop.summary.household_head_ages.values()) >= 18, "Check failed. Min head of household age is younger than 18 years old."
    print("Check passed. All heads of households are at least 18 years old.")

    # check household head age count summary
    assert sum(pop.summary.household_head_age_count.values()) == len(pop.summary.household_sizes), "Check on count of household head ages failed."
    print("Check passed. The count of household head ages matches the number of households created.\n")

    # check ltcf summary
    assert sum(pop.summary.ltcf_sizes.values()) > 0, "Check failed. No people placed in ltcfs."
    print("Check passed. Ltcfs created.")

    # count only LTCF residents
    ltcf_sizes_res = pop.get_ltcf_sizes(keys_to_exclude=['snf_staff'])
    assert sum(ltcf_sizes_res.values()) < sum(pop.summary.ltcf_sizes.values()), "Check failed. Ltcf residents is greater than or equal to all people in ltcfs."
    print("Check passed. Ltcf residents created separately.")

    # check that those living in households or LTCFs account for the entire expected population
    assert sum(pop.summary.household_sizes.values()) + sum(ltcf_sizes_res.values()) == pop.n, f"Check failed. Population size is {pop.n} and the sum of people generated living in households and ltcfs is {sum(pop.summary.household_sizes.values()) + sum(ltcf_sizes_res.values())}."
    print("Check passed. Everyone lives either in a household or ltcf.")

    # count only LTCF staff
    ltcf_sizes_staff = pop.get_ltcf_sizes(keys_to_exclude=['snf_res'])
    assert sum(ltcf_sizes_res.values()) + sum(ltcf_sizes_staff.values()) == sum(pop.summary.ltcf_sizes.values()), "Check failed. The sum of ltcf residets and staff counted separately does not equal the count of them together."
    print("Check passed. Ltcf staff created separately.\n")

    # check enrollment count by age
    assert sum(pop.summary.enrollment_by_age.values()) > 0, f"Check failed. Student enrollment is less than or equal to 0 ({sum(pop.enrollment_by_age.values())})."
    print("Check passed. Student enrollment count by age exists and is greater than 0.")

    # check enrollment rates by age
    enrollment_rates_by_age = pop.enrollment_rates_by_age  # a property rather than stored data so make a copy here
    assert 0 < enrollment_rates_by_age[10] <= 1., f"Check failed. Enrollment rate for age 10 is less than or equal to 0 ({enrollment_rates_by_age[10]}."
    print(f"Check passed. Enrollment rate for age 10 is {enrollment_rates_by_age[10] * 100:.2f}%.\n")

    # check employment rates by age
    employment_rates_by_age = pop.employment_rates_by_age  # a property rather than stored data so make a copy here
    assert 0 < employment_rates_by_age[25] <= 1., f"Check failed. Employment rate for age 25 is less than or equal to 0 ({employment_rates_by_age[25]})."
    print(f"Check passed. Employment rate for age 25 is {employment_rates_by_age[25] * 100:.2f}%.")

    # check workplace sizes
    assert sum(pop.summary.workplace_sizes.values()) > 0, "Check failed. Sum of workplace sizes is less than or equal to 0."
    print("Workplace sizes exists in pop object and is a dictionary by workplace id (wpid).")

    workplace_size_brackets = sp.get_workplace_size_brackets(**pop.loc_pars)

    # check that bins and bin labels can be made
    workplace_size_bins = sp.get_bin_edges(workplace_size_brackets)
    assert len(workplace_size_bins) >= 2, "Check failed. workplace size bins contains the limits for less than one bin."
    print(f"Check passed. There are {len(workplace_size_bins) - 1} workplace size bins.")

    # check that bin labels are all strings
    workplace_size_bin_labels = sp.get_bin_labels(workplace_size_brackets)
    label_types = list(set([type(bl) for bl in workplace_size_bin_labels]))

    assert len(label_types) == 1, f"Check failed. There is more than one type for the workplace size bin labels generated."
    print("Check passed. There is only one type for workplace size bin labels generated.")

    assert isinstance(workplace_size_bin_labels[0], str), f"Check failed. Bin labels are not strings."
    print("Check passed. Bin labels are strings.")

    workplace_size_dist = sp.get_generated_workplace_size_distribution(pop.summary.workplace_sizes, workplace_size_bins)
    expected_workplace_size_dist = sp.norm_dic(sp.get_workplace_size_distr_by_brackets(sp.settings_config.datadir, state_location=pop.state_location, country_location=pop.country_location))
    if expected_workplace_size_dist[0] > 0:
        assert workplace_size_dist[0] > 0, f"Check failed. Expected some workplaces to be created in the smallest bin size but there are none in this bin."
        print("Check passed for workplaces in the smallest bin.")


def test_contact_matrices_used():
    """
    Test that the contact matrices used in generation are left unmodified. The
    workplaces module instead should modify a copy of the workplace contact
    matrix and leave the original contact matrix as is. This means the matrices
    added to the pop object as a summary should match expected data.
    """
    sp.logger.info("Test that the contact matrices used in generation match the expected data.")
    pop = sp.Pop(**pars)

    expected_contact_matrix_dic = sp.get_contact_matrix_dic(sp.settings_config.datadir, sheet_name=pop.sheet_name)
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

    expected_contact_matrix_dic = sp.get_contact_matrix_dic(sp.settings_config.datadir, sheet_name=test_pars.sheet_name)

    # check that the correct contact matrices are used in population generation
    for k in expected_contact_matrix_dic.keys():
        err_msg = f"Check failed for contact setting {k}."
        np.testing.assert_array_equal(pop.contact_matrix_dic[k], expected_contact_matrix_dic[k], err_msg=err_msg)

    print(f"Check passed. {test_pars.sheet_name} contact matrices used in population generation.")


def test_get_contact_matrix_dic_error_handling():
    """Test error handling for sp.get_contact_matrix_dic()."""
    with pytest.raises(RuntimeError) as excinfo:
        sp.get_contact_matrix_dic(sp.settings_config.datadir, sheet_name="notexist")
    assert "Data unavailable for the location specified" in str(excinfo.value), \
        "Error message for non existent sheet should be meaningful"



if __name__ == '__main__':

    test_summary_in_generation()
    test_contact_matrices_used()
    test_change_sheet_name()
    test_get_contact_matrix_dic_error_handling()
