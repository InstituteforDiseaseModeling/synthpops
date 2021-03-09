"""
Test that errors are caught when expected in plotting methods.
"""
import sciris as sc
import synthpops as sp
import covasim as cv
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean as cmo
import cmasher as cmr
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


def test_calculate_contact_matrix_errors():
    """
    Test that synthpops.plotting.calculate_contact_matrix raises an error when
    density_or_frequency is neither 'density' nor 'frequency'.
    """
    sp.logger.info("Catch ValueError when density_or_frequency is not 'density' or 'frequency'.")
    pop = sp.Pop(**pars)
    with pytest.raises(ValueError):
        pop.plot_contacts(**sc.objdict(density_or_frequency='neither'))


def test_catch_pop_type_errors():
    """
    Test that synthpops.plotting methods raise error when pop type is not in
    sp.Pop, dict, or cv.people.People.
    """
    sp.logger.info("Catch NotImplementedError when pop type is invalid.")
    pop = list()

    with pytest.raises(ValueError):
        sp.plot_ages(pop)
    with pytest.raises(ValueError):
        sp.plot_household_sizes(pop)
    with pytest.raises(ValueError):
        sp.plot_ltcf_resident_sizes(pop)
    with pytest.raises(ValueError):
        sp.plot_enrollment_rates_by_age(pop)
    with pytest.raises(ValueError):
        sp.plot_employment_rates_by_age(pop)
    with pytest.raises(ValueError):
        sp.plot_school_sizes(pop)
    with pytest.raises(ValueError):
        sp.plot_workplace_sizes(pop)


if __name__ == '__main__':

    test_calculate_contact_matrix_errors()