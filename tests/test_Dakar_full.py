"""Test Dakar location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings


default_nbrackets = sp.settings.nbrackets

pars = sc.objdict(
    n                               = settings.pop_sizes.small,
    rand_seed                       = 0,

    household_method                = 'fixed_ages',
    location                        = 'Dakar',
    state_location                  = 'Dakar',
    country_location                = 'Senegal',
    use_default                     = False,
)


def test_Dakar():
    """."""

    pop = sp.Pop(**pars)

    print(pop.summarize())

    # print(pop.information)
    # pop.plot_household_sizes()  # update household sizes data to go up to 50 - make sure to calculate these using the household weights and not just pure counts
    # pop.plot_ages()
    # pop.plot_enrollment_rates_by_age()
    # pop.plot_employment_rates_by_age()
    # pop.plot_workplace_sizes()
    pop.plot_household_head_ages_by_size()  # update the household head age by size matrix to go up to 50
    # pop.plot_contacts(layer='H', density_or_frequency='frequency', logcolors_flag=1, aggregate_flag=1)  # test other options
    sp.set_location_defaults()

if __name__ == '__main__':

    test_Dakar()
    plt.show()