"""Example of using method to smooth out age distribution."""

import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean


mplt.rcParams['font.family'] = 'Roboto Condensed'
mplt.rcParams['font.size'] = 12


# parameters to generate a test population
pars = dict(
    country_location = 'usa',
    state_location   = 'Washington',
    # location         = 'seattle_metro',
    location = 'Yakima_County',
    # location = 'Spokane_County',
    # location = 'King_County',
    use_default      = True,
)


if __name__ == '__main__':

    smoothed_age_distr, raw_age_distr = sp.get_smoothed_single_year_age_distr(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'])
    smoothed_age_distr_5, raw_age_distr = sp.get_smoothed_single_year_age_distr(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'], window_length=5)
    smoothed_age_distr_7, raw_age_distr = sp.get_smoothed_single_year_age_distr(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'], window_length=7)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    cmap = mplt.cm.get_cmap('cmo.curl_r')

    age_range = np.array(sorted(smoothed_age_distr.keys()))

    s = dict()

    s[3] = np.array([smoothed_age_distr[a] for a in age_range])
    s[5] = np.array([smoothed_age_distr_5[a] for a in age_range])
    s[7] = np.array([smoothed_age_distr_7[a] for a in age_range])
    r = np.array([raw_age_distr[a] for a in age_range])

    ax.plot(age_range, r, color=cmap(0.15), marker='o', markerfacecolor=cmap(0.27), markersize=3, markeredgewidth=1, alpha=0.65, label='Raw')

    for ns, si in enumerate(sorted(s.keys())):
        ax.plot(age_range, s[si], color=cmap(0.95 - ns * 0.15), marker='o', markerfacecolor=cmap(0.85 - ns * 0.15), markeredgewidth=1, markersize=3, alpha=.75, label=f'Smoothing window = {si}')

        print(si, sum(s[si]), s[si].min())

    leg = ax.legend(loc=1)
    leg.draw_frame(False)
    ax.set_xlim(age_range[0], age_range[-1])
    ax.set_ylim(bottom=0.)
    ax.set_xlabel('Age')
    ax.set_ylabel('Distribution (%)')
    ax.set_title(f"Smoothing Binned Age Distribution: {pars['location'].replace('_', ' ').replace('-', ' ')}")

    plt.show()
