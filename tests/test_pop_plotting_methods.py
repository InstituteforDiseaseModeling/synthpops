"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt


# parameters to generate a test population
pars = dict(
    n                               = 10e3,
    rand_seed                       = 123,
    max_contacts                    = None,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    with_industry_code              = 0,
    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    use_two_group_reduction         = 1,
    with_school_types               = 1,

    average_LTCF_degree             = 20,
    ltcf_staff_age_min              = 20,
    ltcf_staff_age_max              = 60,

    school_mixing_type              = {'pk-es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
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


def test_plot_age_distribution_comparison():
    """
    Test that the age comparison plotting method in sp.Pop class works.
    """
    sp.logger.info("Test that the age comparison plotting method in sp.Pop class works.")

    pop = sp.Pop(**pars)

    fig, ax = pop.plot_age_distribution_comparison()
    return fig, ax

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    fig, ax = test_plot_age_distribution_comparison()
    plt.show()