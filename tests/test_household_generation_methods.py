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
import settings

mplt.rcParams['font.family'] = 'Roboto Condensed'
mplt.rcParams['font.size'] = 8


# parameters to generate a test population
pars = dict(
    n                               = settings.pop_sizes.small,
    rand_seed                       = 123,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    smooth_ages                     = False,
    household_method                = 'infer_ages',

    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    with_school_types               = 1,

    school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with

)


# Todo: pull in new plotting methods directly on pop object
def test_original_household_method(do_show=False):
    sp.logger.info("Generating households with the infer_ages method.")

    test_pars = sc.dcp(pars)
    test_pars['household_method'] = 'infer_ages'
    pop = sp.Pop(**test_pars)
    popdict = pop.to_dict()

    datadir = sp.datadir
    fig, ax = plot_age_dist(datadir, popdict, test_pars, do_show, test_pars['household_method'])

    if do_show:
        plt.show()

    return pop


# Todo: pull in new plotting methods directly on pop object
def test_fixed_ages_household_method(do_show=False):
    sp.logger.info("Generating households with the fixed_ages method.")

    test_pars = sc.dcp(pars)
    test_pars['n'] = settings.pop_sizes.large
    test_pars['household_method'] = 'fixed_ages'
    pop = sp.Pop(**test_pars)
    popdict = pop.to_dict()

    datadir = sp.datadir
    fig, ax = plot_age_dist(datadir, popdict, test_pars, do_show, test_pars['household_method'])

    if do_show:
        plt.show()

    return fig, ax


# Todo: pull in new plotting methods directly on pop object
def test_smoothed_and_fixed_ages_household_method(do_show=False):
    sp.logger.info("Generating households with the fixed_ages and smoothed_ages methods.")

    test_pars = sc.dcp(pars)
    test_pars['n'] = settings.pop_sizes.large
    test_pars['location'] = 'Spokane_County'
    test_pars['household_method'] = 'fixed_ages'
    test_pars['smooth_ages'] = True
    test_pars['window_length'] = 7  # window for averaging the age distribution
    pop = sp.Pop(**test_pars)
    popdict = pop.to_dict()

    datadir = sp.datadir
    fig, ax = plot_age_dist(datadir, popdict, test_pars, do_show, test_pars['household_method'])

    if do_show:
        plt.show()

    return fig, ax


# duplicate / early version of plotting method now available
def plot_age_dist(datadir, pop, pars, do_show, prefix):
    sp.logger.info("Plot the expected age distribution and the generated age distribution.")

    age_brackets = sp.get_census_age_brackets(datadir, country_location=pars['country_location'],
                                              state_location=pars['state_location'], location=pars['location'])
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    if pars['smooth_ages']:
        expected_age_distr = sp.get_smoothed_single_year_age_distr(datadir, location=pars['location'],
                                                                   state_location=pars['state_location'],
                                                                   country_location=pars['country_location'],
                                                                   window_length=pars['window_length'])

    else:
        expected_age_distr = sp.get_smoothed_single_year_age_distr(datadir, location=pars['location'],
                                                                   state_location=pars['state_location'],
                                                                   country_location=pars['country_location'],
                                                                   window_length=1)

    gen_age_count = dict.fromkeys(expected_age_distr.keys(), 0)

    for i, person in pop.items():
        gen_age_count[person['age']] += 1

    gen_age_distr = sp.norm_dic(gen_age_count)

    fig, ax = sppl.plot_array([v * 100 for v in expected_age_distr.values()],
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
