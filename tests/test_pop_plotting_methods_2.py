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


def test_plot_household_sizes_dist(do_show=False, do_save=False):
    """
    Test that the household sizes comparison plotting method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test the household sizes comparison plotting method works with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_household_sizes_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    fig, ax = pop.plot_household_sizes(**kwargs)
    # fig, ax = pop.plot_household_sizes()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    sp.logger.info("Test the household sizes comparison plotting method with a population dictionary.")
    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_household_sizes_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_household_sizes(popdict, **kwargs)
    # fig2, ax2 = sp.plot_household_sizes(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    sp.logger.info("Test the household sizes plotting method without comparison.")
    kwargs.comparison = False
    fig3, ax3 = pop.plot_household_sizes(**kwargs)
    assert isinstance(fig3, mplt.figure.Figure), 'Check 3 failed.'
    print('Check passed. Plotting without comparison.')

    return fig, ax, pop


def test_plot_ltcf_resident_sizes(do_show=False, do_save=False):
    """
    Test that the long term care facility resident sizes comparison plotting
    method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test the long term care facility resident sizes comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_ltcf_resident_sizes_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    fig, ax = pop.plot_ltcf_resident_sizes(**kwargs)
    # fig, ax = pop.plot_ltcf_resident_sizes()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    sp.logger.info("Test the long term care facility resident sizes comparison plotting method with a population dictionary.")
    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_ltcf_resident_sizes_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_ltcf_resident_sizes(popdict, **kwargs)
    # fig2, ax2 = sp.plot_ltcf_resident_sizes(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    sp.logger.info("Test the long term care facility resident sizes plotting method without comparison.")
    kwargs.comparison = False
    fig3, ax3 = pop.plot_ltcf_resident_sizes(**kwargs)
    assert isinstance(fig3, mplt.figure.Figure), 'Check 3 failed.'
    print('Check passed. Plotting without comparison.')

    return fig, ax, pop


def test_plot_enrollment_rates_by_age(do_show=False, do_save=False):
    """
    Test that the enrollment rates comparison plotting method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test the enrollment rates comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_enrollment_rates_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    print(pars.n)
    fig, ax = pop.plot_enrollment_rates_by_age(**kwargs)
    # fig, ax = pop.plot_enrollment_rates_by_age()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    sp.logger.info("Test the enrollment rates comparison plotting method with a population dictionary.")
    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_enrollment_rates_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_enrollment_rates_by_age(popdict, **kwargs)
    # fig2, ax2 = sp.plot_enrollment_rates_by_age(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    sp.logger.info("Test the enrollment rates plotting method without comparison.")
    kwargs.comparison = False
    fig3, ax3 = pop.plot_enrollment_rates_by_age(**kwargs)
    assert isinstance(fig3, mplt.figure.Figure), 'Check 3 failed.'
    print('Check passed. Plotting without comparison.')

    return fig, ax, pop


def test_plot_employment_rates_by_age(do_show=False, do_save=False):
    """
    Test that the employment rates comparison plotting method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test the employment rates comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    print(pars.n)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_employment_rates_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    fig, ax = pop.plot_employment_rates_by_age(**kwargs)
    # fig, ax = pop.plot_employment_rates_by_age()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    sp.logger.info("Test the employment rates comparison plotting method with a population dictionary.")
    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_employment_rates_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_employment_rates_by_age(popdict, **kwargs)
    # fig2, ax2 = sp.plot_employment_rates_by_age(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    sp.logger.info("Test the employment rates plotting method without comparison.")
    kwargs.comparison = False
    fig3, ax3 = pop.plot_employment_rates_by_age(**kwargs)
    assert isinstance(fig3, mplt.figure.Figure), 'Check 3 failed.'
    print('Check passed. Plotting without comparison.')

    return fig, ax, pop


def test_plot_workplace_sizes(do_show=False, do_save=False):
    """
    Test that the workplace sizes comparison plotting
    method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test the workplace sizes comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_workplace_sizes_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    fig, ax = pop.plot_workplace_sizes(**kwargs)
    # fig, ax = pop.plot_workplace_sizes()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    sp.logger.info("Test the workplace sizes comparison plotting method with a population dictionary.")
    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_workplace_sizes_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_workplace_sizes(popdict, **kwargs)
    # fig2, ax2 = sp.plot_workplace_sizes(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    sp.logger.info("Test the workplace sizes plotting method without comparison.")
    kwargs.comparison = False
    fig3, ax3 = pop.plot_ages(**kwargs)
    assert isinstance(fig3, mplt.figure.Figure), 'Check 3 failed.'
    print('Check passed. Plotting without comparison.')

    return fig, ax, pop


if __name__ == '__main__':

    # run as main and see the examples in action!

    fig0, ax0, pop0 = test_plot_household_sizes_dist(do_show=True)
    fig1, ax1, pop1 = test_plot_ltcf_resident_sizes(do_show=True)
    fig2, ax2, pop2 = test_plot_enrollment_rates_by_age(do_show=True)
    fig3, ax3, pop3 = test_plot_employment_rates_by_age(do_show=True)
    fig4, ax4, pop4 = test_plot_workplace_sizes(do_show=True)