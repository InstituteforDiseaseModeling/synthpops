"""Plot figure of the age demographics from the 4th schools report."""
import numpy as np
import sciris as sc
import synthpops as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmasher as cmr
import cmocean as cmo
import os


mplt.rcParams['font.family'] = 'Roboto Condensed'  # Nicer font being set
mplt.rcParams['font.size'] = 16


# population parameters
pars = sc.objdict(
    n                               = 225e3,  # population size
    rand_seed                       = 123,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default=True,

    household_method                = 'fixed_ages',
    smooth_ages                     = 1,

    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    with_school_types               = 1,


    school_mixing_type              = {'pk': 'age_and_class_clustered',
                                       'es': 'age_and_class_clustered',
                                       'ms': 'age_and_class_clustered',
                                       'hs': 'random', 'uv': 'random'},
    average_student_teacher_ratio   = 20,  # location specific
    average_student_all_staff_ratio = 11,  # from Seattle Public Schools
)

kwargs = sc.dcp(pars)  # add to plotting parameters

colors = dict(
    seattle_metro   = 'tab:green',
    Spokane_County  = 'tab:purple',
    Franklin_County = 'tab:red',
    Island_County   = 'tab:blue',
)

labels = dict(
    seattle_metro   = 'Seattle',
    Spokane_County  = 'Spokane',
    Franklin_County = 'Franklin',
    Island_County   = 'Island',
)


locations = ['seattle_metro', 'Spokane_County', 'Franklin_County', 'Island_County']

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.15)

for n, location in enumerate(locations):

    loc_pars = dict(location=location, state_location=pars.state_location, country_location=pars.country_location,
                    datadir=sp.datadir, nbrackets=18)
    age_bracket_distr = sp.read_age_bracket_distr(**loc_pars)
    age_brackets = sp.get_census_age_brackets(**loc_pars)

    x = np.arange(len(age_brackets))
    y = [age_bracket_distr[b] * 100 for b in age_brackets]
    bin_labels = [f"{age_brackets[b][0]}-{age_brackets[b][-1]}" for b in age_brackets]

    ax.plot(x, y, color=colors[location], label=labels[location], marker='o', markerfacecolor=colors[location],
            markeredgecolor='white', markeredgewidth=2, markersize=8,)
leg = ax.legend(loc=1)

ax.set_xlim(-0.2, x[-1] + 0.2)
ax.set_ylim(0, 10)
ax.set_xlabel('Age Group', fontsize=16)
ax.set_ylabel('(%)', fontsize=16)

ticks = x
ticklabels = bin_labels
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, rotation=40, verticalalignment='center_baseline')
ax.set_title('Age Distribution', fontsize=20)

ax.tick_params(labelsize=15)
dpi = 300
fig.savefig('AgeDistributionsbyLocation.png', format='png', dpi=dpi)
