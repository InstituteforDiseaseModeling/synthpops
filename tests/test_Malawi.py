"""Test Malawi location works and plot the demographics and contact networks."""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest


default_nbrackets = sp.settings.nbrackets

pars = sc.objdict(
    n                = settings.pop_sizes.small,
    rand_seed        = 0,

    household_method = 'fixed_ages',
    smooth_ages      = 1,

    country_location = 'Malawi',
    sheet_name       = 'Zambia',
    use_default      = True,
)

if __name__ == '__main__':
    sp.set_location_defaults(country_location="Senegal")
    pop = sp.Pop(**pars)
    print(pop.summarize())
    pop.plot_ages()
    pop.plot_enrollment_rates_by_age()
    sp.set_location_defaults()
    plt.show()
