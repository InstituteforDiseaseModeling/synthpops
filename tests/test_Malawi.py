"""Test Malawi location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest


pars = sc.objdict(
    n                = settings.pop_sizes.small,
    rand_seed        = 123,

    household_method = 'fixed_ages',
    smooth_ages      = 1,

    country_location = 'Malawi',
    sheet_name       = 'Zambia',  # no malawi contact patterns in prem et al. 2017
    use_default      = True,
)


def test_Malawi():
    """Test Malawi population constructed."""
    sp.logger.info("Test that Malawi contact networks can be made. Not a guarantee that the population made matches age mixing patterns well yet.")

    # reset the default location to pull other data
    sp.set_location_defaults(country_location="Senegal")
    # make a basic population
    pop = sp.Pop(**pars)
    assert pop.country_location == 'Malawi', "population location information is not set to Malawi"
    sp.reset_default_settings()  # reset defaults


def pop_exploration():
    sp.logger.info("Exploration of the Malawi population generation with default methods and missing data filled in with Senegal data")
    sp.set_location_defaults(country_location="Senegal")
    pop = sp.Pop(**pars)
    print(pop.summarize())
    pop.plot_ages()
    pop.plot_enrollment_rates_by_age()
    sp.set_location_defaults()
    plt.show()
    sp.reset_default_settings()  # reset defaults


if __name__ == '__main__':
    test_Malawi()
    pop_exploration()
