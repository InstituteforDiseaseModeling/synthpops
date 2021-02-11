"""Plot figure of the school sizes from the 4th schools report."""
import numpy as np
import sciris as sc
import synthpops as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmasher as cmr
import cmocean as cmo
import os


mplt.rcParams['font.family'] = 'Roboto Condensed'  # Nicer font being set
mplt.rcParams['font.size'] = 15

# population parameters
pars = sc.objdict(
    n                               = 225e3,
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

edgecolors = dict(
    seattle_metro   = '#217821',
    Spokane_County  = '#6e4196',
    Franklin_County = '#951b1c',
    Island_County   = '#185a88',
)

labels = dict(
    seattle_metro   = 'Seattle',
    Spokane_County  = 'Spokane',
    Franklin_County = 'Franklin',
    Island_County   = 'Island',
)


locations = ['seattle_metro', 'Spokane_County', 'Franklin_County', 'Island_County']

school_types = ['pk', 'es', 'ms', 'hs']
school_type_labels = dict(
    pk='Preschool',
    es='Elementary School',
    ms='Middle School',
    hs='High School'
)

figs, axs = [], []

for ns, school_type in enumerate(school_types):
    fig, ax = plt.subplots(len(locations), 1, figsize=(6., 11.))
    fig.subplots_adjust(left=0.12, right=0.965, top=0.96, bottom=0.1, hspace=0.8)
    figs.append(fig)
    axs.append(ax)

for n, location in enumerate(locations):

    loc_pars = dict(location=location, state_location=pars.state_location, country_location=pars.country_location,
                    datadir=sp.datadir)

    school_size_brackets = sp.get_school_size_brackets(**loc_pars)

    bins = [school_size_brackets[0][0]] + [school_size_brackets[b][-1] + 1 for b in school_size_brackets]
    bin_labels = [f"{school_size_brackets[b][0]}-{school_size_brackets[b][-1]}" for b in school_size_brackets]

    expected_school_size_distr = sp.get_school_size_distr_by_type(**loc_pars)

    x = np.arange(len(school_size_brackets))

    for ns, school_type in enumerate(school_types):

        sorted_bins = sorted(expected_school_size_distr[school_type].keys())

        axs[ns][n].bar(x, [expected_school_size_distr[school_type][b] * 100 for b in sorted_bins],
                       color=colors[location], edgecolor=edgecolors[location],
                       linewidth=2,
                       label=labels[location])

        axs[ns][n].set_title(labels[location], fontsize=20)
        axs[ns][n].set_xlim(-0.6, x[-1] + 0.6)
        axs[ns][n].set_ylim(0, 100)
        axs[ns][n].set_xticks(x)
        axs[ns][n].set_xticklabels(bin_labels, rotation=50, fontsize=15, verticalalignment='center_baseline')
        yticks = np.arange(0, 101, 25)
        yticklabels = [str(i) for i in yticks]
        axs[ns][n].set_yticks(yticks)
        axs[ns][n].set_yticklabels(yticklabels)
        axs[ns][n].set_ylabel('(%)', fontsize=16)
        if n == 0:
            axs[ns][n].text(x[-1] + 0.6, 118, school_type_labels[school_type], fontsize=20, verticalalignment='top', horizontalalignment='right')

        if n == len(locations) - 1:
            axs[ns][n].set_xlabel('School Sizes', fontsize=18)

dpi = 300
for nf, fig in enumerate(figs):

    fig.savefig(f"SchoolSize_{school_type_labels[school_types[nf]].replace(' ', '_')}.png", format='png', dpi=dpi)
