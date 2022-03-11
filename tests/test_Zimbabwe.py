"""Test Zimbabwe location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest


pars = sc.objdict(
    n                = settings.pop_sizes.small_medium,
    rand_seed        = 0,

    household_method = 'fixed_ages',
    smooth_ages      = 1,

    country_location = 'Zimbabwe',
    sheet_name       = 'Zimbabwe',
    use_default      = True,
    with_school_types = 1,
)


def test_Zimbabwe():
    """Test Zimbabwe population constructed."""
    sp.logger.info("Test that Zimbabwe contact networks can be made. Not a guarantee that the population made matches age mixing patterns well yet.")

    # reset the default location to pull other data
    sp.set_location_defaults(country_location="Senegal")
    # make a basic population
    pop = sp.Pop(**pars)
    assert pop.country_location == 'Zimbabwe', "population location information is not set to Zimbabwe"
    sp.reset_default_settings()  # reset defaults


def pop_exploration():
    sp.logger.info("Exploration of the Zimbabwe population generation with default methods and missing data filled in with Senegal data")
    sp.set_location_defaults(country_location="Senegal")
    pop = sp.Pop(**pars)
    print(pop.summarize())
    pop.plot_ages()
    pop.plot_household_sizes()
    pop.plot_enrollment_rates_by_age()
    pop.plot_contacts(layer='H', density_or_frequency='density', logcolors_flag=0, title_prefix="Zimbabwe Age Mixing")
    pop.plot_school_sizes(with_school_types=1)
    pop.plot_employment_rates_by_age()
    pop.plot_workplace_sizes()
    sp.reset_default_settings()  # reset defaults
    plt.show()


if __name__ == '__main__':
    test_Zimbabwe()
    pop_exploration()
