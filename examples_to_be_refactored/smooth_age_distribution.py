"""Example of using method to smooth out age distribution."""

import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmasher as cmr
import cmocean


mplt.rcParams['font.family'] = 'Roboto Condensed'
mplt.rcParams['font.size'] = 12


# parameters to generate a test population
pars = dict(
    country_location = 'usa',
    state_location   = 'Washington',
    location       = 'seattle_metro',
    # location         = 'Spokane_County',
    use_default      = True,
)


def smooth_binned_age_distribution(pars, do_show=False):
    sp.logger.info(f"Smoothing out age distributions with moving averages.")

    s = dict()
    # raw_age_bracket_distr = sp.read_age_bracket_distr(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'])
    raw_age_distr = sp.get_smoothed_single_year_age_distr(sp.datadir, location=pars['location'],
                                                          state_location=pars['state_location'],
                                                          country_location=pars['country_location'],
                                                          window_length=1)
    age_brackets = sp.get_census_age_brackets(sp.datadir, country_location=pars['country_location'],
                                              state_location=pars['state_location'], location=pars['location'])
    max_age = age_brackets[max(age_brackets.keys())][-1]

    age_range = np.arange(max_age + 1)

    for si in np.arange(3, 8, 2):

        smoothed_age_distr = sp.get_smoothed_single_year_age_distr(sp.datadir, location=pars['location'],
                                                                   state_location=pars['state_location'],
                                                                   country_location=pars['country_location'],
                                                                   window_length=si)
        s[si] = np.array([smoothed_age_distr[a] for a in age_range])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    cmap = mplt.cm.get_cmap('cmr.ember')

    if len(s) > 3:
        cmap1 = cmr.get_sub_cmap('cmr.rainforest', 0.13, 0.85)
        cmap2 = cmr.get_sub_cmap('cmr.rainforest', 0.20, 0.92)
    else:
        cmap1 = cmr.get_sub_cmap('cmr.rainforest', 0.18, 0.68)
        cmap2 = cmr.get_sub_cmap('cmr.rainforest', 0.25, 0.75)

    delta = 1 / (len(s) - 1)

    age_range = np.array(sorted(smoothed_age_distr.keys()))

    r = np.array([raw_age_distr[a] for a in age_range])

    ax.plot(age_range, r, color=cmap(0.55), marker='o', markerfacecolor=cmap(0.65), markersize=3, markeredgewidth=1, alpha=0.65, label='Raw')

    for ns, si in enumerate(sorted(s.keys())):
        ax.plot(age_range, s[si], color=cmap1(ns * delta), marker='o', markerfacecolor=cmap2(ns * delta), markeredgewidth=1, markersize=3, alpha=.75, label=f'Smoothing window = {si}')

    leg = ax.legend(loc=3)
    leg.draw_frame(False)
    ax.set_xlim(age_range[0], age_range[-1])
    ax.set_ylim(bottom=0.)
    ax.set_xlabel('Age')
    ax.set_ylabel('Distribution (%)')
    ax.set_title(f"Smoothing Binned Age Distribution: {pars['location'].replace('_', ' ').replace('-', ' ')}")

    if do_show:
        plt.show()

    return fig, ax


if __name__ == '__main__':

    fig, ax = smooth_binned_age_distribution(pars, do_show=True)
