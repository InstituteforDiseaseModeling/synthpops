"""Test Spokane location works."""
import sciris as sc
import synthpops as sp
import matplotlib.pyplot as plt


pars = dict(
    n                               = 20000,
    rand_seed                       = 123,
    max_contacts                    = None,
    location                        = 'Spokane_County',
    state_location                  = 'Washington',
    country_location                = 'usa',
    use_default                     = False,

    with_industry_code              = 0,
    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    use_two_group_reduction         = 1,
    with_school_types               = 1,

    average_LTCF_degree             = 20,
    ltcf_staff_age_min              = 20,
    ltcf_staff_age_max              = 60,

    school_mixing_type              = 'age_and_class_clustered',
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


def test_Spokane():
    """Test that a Dakar population can be created with the basic SynthPops API."""
    sp.logger.info("Test that a Dakar population can be created with the basic SynthPops API.")
    pop = sp.make_population(**pars)
    age_distr = sp.read_age_bracket_distr(sp.datadir, country_location='usa', state_location='Washington', location='seattle_metro')
    assert len(age_distr) == 20, f'Check failed, len(age_distr): {len(age_distr)}'  # will remove if this passes in github actions test

    sp.set_location_defaults('defaults')  # Reset default values after this test is complete.
    return pop


"""
Notes:

data missing:

ltcf resident sizes -- copied facility sizes from Seattle, King County to Spokane County.

"""


if __name__ == '__main__':
    T = sc.tic()
    pop = test_Spokane()
    sc.toc(T)
    print(f"Spokane County population of size {pars['n']} made.")
