"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import covasim as cv
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean as cmo
import cmasher as cmr
import pytest
import settings


# parameters to generate a test population
pars = dict(
    n                       = settings.pop_sizes.small_medium,
    rand_seed               = 123,

    smooth_ages             = True,

    with_facilities         = 1,
    with_non_teaching_staff = 1,
    with_school_types       = 1,

    school_mixing_type      = {'pk': 'age_and_class_clustered',
                               'es': 'age_and_class_clustered',
                               'ms': 'age_and_class_clustered',
                               'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
)
pars = sc.objdict(pars)


def test_plot_enrollment_rates_by_age(do_show=False, do_save=False):
    """
    Test that the enrollment rates comparison plotting method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the enrollment rates comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_pop_ages_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    fig, ax = pop.plot_enrollment_rates_by_age(**kwargs)
    # fig, ax = pop.plot_enrollment_rates_by_age()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_popdict_enrollment_rates_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_enrollment_rates_by_age(popdict, **kwargs)
    # fig2, ax2 = sp.plot_enrollment_rates_by_age(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')
    return fig, ax, pop


def test_plot_employment_rates_by_age(do_show=False, do_save=False):
    """
    Test that the employment rates comparison plotting method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the employment rates comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_pop_ages_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    fig, ax = pop.plot_employment_rates_by_age(**kwargs)
    # fig, ax = pop.plot_employment_rates_by_age()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_popdict_enrollment_rates_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_employment_rates_by_age(popdict, **kwargs)
    # fig2, ax2 = sp.plot_employment_rates_by_age(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')
    return fig, ax, pop


if __name__ == '__main__':

    # run as main and see the examples in action!

    fig0, ax0, pop0 = test_plot_enrollment_rates_by_age(do_show=True)
    fig1, ax1, pop1 = test_plot_employment_rates_by_age(do_show=True)
