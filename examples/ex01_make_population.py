"""
Make a population using synthpops.
"""
import synthpops as sp
import matplotlib.pyplot as plt

pars = dict(
    n                = 10e3,
    rand_seed        = 123,
    smooth_ages      = 1,
    household_method = 'fixed_ages',
)

pop = sp.Pop(**pars)  # generate networked population

fig1, ax1 = pop.plot_ages()  # plot age distribution and comparsion to data
fig2, ax2 = pop.plot_enrollment_rates_by_age()  # plot enrollment rates by age and comparison to data

plt.show()