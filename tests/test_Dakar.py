"""Test Dakar location works with basic synthpops methodology."""
import sciris as sc
import synthpops as sp

sp.set_nbrackets(18)  # Dakar age distributions available are up to 18 age brackets

pars = dict(
    n                               = 20000,
    rand_seed                       = 0,
    max_contacts                    = None,
    location                        = 'Dakar',
    state_location                  = 'Dakar',
    country_location                = 'Senegal',
    use_default                     = False,

    with_industry_code              = 0,
    with_facilities                 = 0,
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


"""
Notes:

This method does not include socioeconomic conditions that are likely
associated with school enrollment.

"""

if __name__ == '__main__':
    T = sc.tic()
    pop = sp.make_population(**pars)
    sc.toc(T)
    print(f"Dakar, Senegal population of size {pars['n']} made.")
