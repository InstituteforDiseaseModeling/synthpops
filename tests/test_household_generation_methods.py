"""
Test that new household generation methods produce populations.
"""

import sciris as sc
import synthpops as sp
import synthpops.plotting as sppl
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmasher as cmr
import cmocean
import pytest

mplt.rcParams['font.family'] = 'Roboto Condensed'
mplt.rcParams['font.size'] = 8


# parameters to generate a test population
pars = dict(
    n                               = 10e3,
    rand_seed                       = 123,

    # need location parameters
    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    smooth_ages                     = False,
    household_method                = 'infer_ages',
    with_facilities                 = 1,

)


def test_original_household_method(do_show=False):
    sp.logger.info("Generating households with the infer_ages method.")

    test_pars = sc.dcp(pars)
    test_pars['household_method'] = 'infer_ages'
    pop = sp.Pop(**test_pars)

    datadir = sp.datadir
    fig, ax = plot_age_dist(datadir, pop, test_pars, do_show, test_pars['household_method'])

    if do_show:
        plt.show()

    return pop


def test_fixed_ages_household_method(do_show=False):
    sp.logger.info("Generating households with the fixed_ages method.")

    test_pars = sc.dcp(pars)
    test_pars['household_method'] = 'fixed_ages'
    pop = sp.Pop(**test_pars)

    datadir = sp.datadir
    fig, ax = plot_age_dist(datadir, pop, test_pars, do_show, test_pars['household_method'])

    if do_show:
        plt.show()

    return fig, ax


def test_smoothed_and_fixed_ages_household_method(do_show=False):
    sp.logger.info("Generating households with the fixed_ages and smoothed_ages methods.")

    test_pars = sc.dcp(pars)
    test_pars['location'] = 'Spokane_County'
    test_pars['household_method'] = 'fixed_ages'
    test_pars['smooth_ages'] = True
    test_pars['window_length'] = 7  # window for averaging the age distribution
    pop = sp.Pop(**test_pars)

    datadir = sp.datadir
    fig, ax = plot_age_dist(datadir, pop, test_pars, do_show, test_pars['household_method'])

    if do_show:
        plt.show()

    return fig, ax


def plot_age_dist(datadir, pop, pars, do_show, prefix):
    sp.logger.info("Plot the expected age distribution and the generated age distribution.")
    loc_pars = pop.loc_pars
    age_brackets = sp.get_census_age_brackets(**loc_pars)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    if pars['smooth_ages']:
        expected_age_distr = sp.get_smoothed_single_year_age_distr(**sc.mergedicts(loc_pars, {'window_length': pars['window_length']}))
    else:
        expected_age_distr = sp.get_smoothed_single_year_age_distr(**sc.mergedicts(loc_pars, {'window_length': 1}))

    gen_age_count = pop.count_pop_ages()
    gen_age_distr = sp.norm_dic(gen_age_count)

    fig, ax = sppl.plot_array([v * 100 for v in expected_age_distr.values()], figname='age_comparison',
                              generated=[v * 100 for v in gen_age_distr.values()], do_show=False, binned=True, prefix=prefix.replace('_', ' '))
    ax.set_xlabel('Ages')
    ax.set_ylabel('Distribution (%)')
    ax.set_ylim(bottom=0)
    ax.set_xlim(-1.5, max(age_by_brackets_dic.keys()) + 1.5)
    ax.set_title(f"Age Distribution of {pars['location'].replace('_', ' ')}: {pars['household_method'].replace('_', ' ')} method")
    fig.set_figheight(4)  # reset the figure size
    fig.set_figwidth(7)

    return fig, ax


if __name__ == '__main__':

    pop = test_original_household_method(do_show=True)

    fig, ax = test_fixed_ages_household_method(do_show=True)

    fig, ax = test_smoothed_and_fixed_ages_household_method(do_show=True)
