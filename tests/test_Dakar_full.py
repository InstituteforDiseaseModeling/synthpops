"""Test Dakar location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest


pars = sc.objdict(
    n                = settings.pop_sizes.medium,
    rand_seed        = 123,

    household_method = 'fixed_ages',
    smooth_ages      = 1,

    location         = 'Dakar',
    state_location   = 'Dakar',
    country_location = 'Senegal',
    sheet_name       = 'Senegal',
    use_default      = False,
)


@pytest.mark.skip
def test_Dakar():
    """Test Dakar population constructed."""
    sp.logger.info("Not a real test yet. To be filled out.")
    pop = sp.Pop(**pars)
    assert pop.location == 'Dakar', 'population location information is not set to Dakar'

    sp.reset_default_settings()  # reset defaults


def pop_exploration():
    sp.logger.info("Exploration of the Dakar population generation with default methods")
    pop = sp.Pop(**pars)
    print(pop.summarize())
    print(pop.information)
    pop.plot_household_sizes()  # update household sizes data to go up to 50 - make sure to calculate these using the household weights and not just pure counts
    pop.plot_ages()
    pop.plot_enrollment_rates_by_age()
    pop.plot_employment_rates_by_age()
    pop.plot_workplace_sizes()
    pop.plot_household_head_ages_by_size()  # update the household head age by size matrix to go up to 50
    pop.plot_contacts(layer='H', density_or_frequency='frequency', logcolors_flag=1, aggregate_flag=1)  # test other options
    pop.plot_contacts(layer='S', density_or_frequency='frequency', logcolors_flag=1, aggregate_flag=1)  # test other options
    pop.plot_contacts(layer='W', density_or_frequency='frequency', logcolors_flag=1, aggregate_flag=1)  # test other options
    plt.show()
    sp.reset_default_settings()  # reset defaults


if __name__ == '__main__':

    test_Dakar()
    pop_exploration()
