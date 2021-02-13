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
pars = sc.objdict(
    n                = settings.pop_sizes.small_medium,
    rand_seed        = 123,

    # need location parameters
    country_location = 'usa',
    state_location   = 'Washington',
    location         = 'seattle_metro',
    use_default      = True,

    smooth_ages      = False,
    household_method = 'infer_ages',

    with_facilities  = 1,

)

kwargs = sc.objdict(
    color_1  = 'mediumseagreen',
    color_2  = '#236a54',
    height   = 4,
    width    = 7,
    fontsize = 9
    )


def test_original_household_method(do_show=False):
    sp.logger.info("Generating households with the infer_ages method.")

    test_pars = sc.dcp(pars)
    test_pars.household_method = 'infer_ages'
    pop = sp.Pop(**test_pars)

    kwargs.update(test_pars)
    kwargs.do_show = do_show
    kwargs.title_prefix = f"Age Distribution of {kwargs.location.replace('_', ' ')}: {kwargs.household_method.replace('_', ' ')} method"

    fig, ax = pop.plot_ages(**kwargs)

    return pop


def test_fixed_ages_household_method(do_show=False):
    sp.logger.info("Generating households with the fixed_ages method.")

    test_pars = sc.dcp(pars)
    test_pars.household_method = 'fixed_ages'
    pop = sp.Pop(**test_pars)

    kwargs.update(test_pars)
    kwargs.do_show = do_show
    kwargs.title_prefix = f"Age Distribution of {kwargs.location.replace('_', ' ')}: {kwargs.household_method.replace('_', ' ')} method"

    fig, ax = pop.plot_ages(**kwargs)

    return fig, ax


def test_smoothed_and_fixed_ages_household_method(do_show=False):
    sp.logger.info("Generating households with the fixed_ages and smoothed_ages methods.")

    test_pars = sc.dcp(pars)
    test_pars.n = settings.pop_sizes.medium
    test_pars.location = 'Spokane_County'
    test_pars.household_method = 'fixed_ages'
    test_pars.smooth_ages = True

    pop = sp.Pop(**test_pars)

    kwargs.update(test_pars)
    kwargs.do_show = do_show
    kwargs.title_prefix = f"Smoothed Age Distribution of {kwargs.location.replace('_', ' ')}: {kwargs.household_method.replace('_', ' ')} method"

    fig, ax = pop.plot_ages(**kwargs)
    return fig, ax


if __name__ == '__main__':

    pop = test_original_household_method(do_show=True)

    fig, ax = test_fixed_ages_household_method(do_show=True)

    fig, ax = test_smoothed_and_fixed_ages_household_method(do_show=True)
