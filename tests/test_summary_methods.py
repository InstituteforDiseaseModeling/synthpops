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
    Test that summaries are produced produced when synthpops generates
    populations.
    """
    sp.logger.info("Test summaries are produced when populations are generated.")

    pop = sp.Pop(**pars)

    assert isinstance(pop.age_count, dict), 'Check failed'
    print(f"Age count summary exists and is a dictionary. The age range is from {min(pop.age_count.keys())} to {max(pop.age_count.keys())} years old.")


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


def test_get_contact_matrix_dic_error_handling():
    """Test error handling for sp.get_contact_matrix_dic()."""
    with pytest.raises(RuntimeError) as excinfo:
        sp.get_contact_matrix_dic(sp.datadir, sheet_name="notexist")
    assert "Data unavailable for the location specified" in str(excinfo.value), \
        "Error message for non existent sheet should be meaningful"



if __name__ == '__main__':

    test_summary_in_generation()
    test_contact_matrices_used()
    test_change_sheet_name()
    test_get_contact_matrix_dic_error_handling()

