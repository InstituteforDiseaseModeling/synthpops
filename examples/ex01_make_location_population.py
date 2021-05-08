"""
Make a population using synthpops for a specified location different from the
default location (seattle_metro).
"""
import synthpops as sp
import matplotlib.pyplot as plt

pars = dict(
    n                 = 10e3,
    rand_seed         = 123,
    location          = 'Spokane_County',
    state_location    = 'Washington',
    country_location  = 'usa',
    smooth_ages       = 1,
    household_method  = 'fixed_ages',
)


pop = sp.Pop(**pars)  # generate contact network
pop.plot_ages()  # plot the age distribution

plt.show()  # show