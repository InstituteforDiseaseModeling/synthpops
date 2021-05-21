"""Test Zimbabwe location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest

pars = sc.objdict(
    n                = settings.pop_sizes.small,
    rand_seed        = 0,

    household_method = 'fixed_ages',
    smooth_ages      = 1,

    country_location = 'Nepal',
    sheet_name       = 'Nepal',
    use_default      = False,
    with_school_types = 1,
)


def test_Nepal():
    """Test Nepal population constructed."""
    sp.logger.info("Test that Nepal contact networks can be made. Not a guarantee that the population made matches age mixing patterns well yet.")

    # make a basic population
    pop = sp.Pop(**pars)
    assert pop.country_location == 'Nepal', "population location information is not set to Malawi"
    sp.reset_default_settings()  # reset defaults so that other tests in parallel are not impacted

def pop_exploration():
    sp.logger.info("Exploration of the Nepal population generation with default methods")
    pop = sp.Pop(**pars)
    print(pop.summarize())
    pop.plot_ages()
    pop.plot_household_sizes()
    pop.plot_enrollment_rates_by_age()
    pop.plot_contacts(layer='H', density_or_frequency='density', logcolors_flag=0, title_prefix="Nepal Age Mixing")
    pop.plot_school_sizes(with_school_types=1)
    pop.plot_employment_rates_by_age()
    pop.plot_workplace_sizes()
    sp.set_location_defaults()
    plt.show()
    # sp.reset_default_settings()  # reset defaults


if __name__ == '__main__':
    test_Nepal()
    # pop_exploration()





