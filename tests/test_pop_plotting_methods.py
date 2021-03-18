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
pars = sc.objdict(
    n                       = settings.pop_sizes.small,
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


def test_plot_ages(do_show=False, do_save=False):
    """
    Test that the age comparison plotting method in sp.Pop class works.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the age comparison plotting method with sp.Pop object.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_pop_ages_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    fig, ax = pop.plot_ages(**kwargs)
    # fig, ax = pop.plot_ages()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_popdict_ages_{kwargs.location}_popdict"
    kwargs.do_show = False
    fig2, ax2 = sp.plot_ages(popdict, **kwargs)
    # fig2, ax2 = sp.plot_ages(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')
    return fig, ax, pop


def test_plot_with_cvpeople(do_show=False, do_save=False):
    """
    Test plotting method works on covasim.people.People object.

    Notes:
        With this pop type, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the age comparison plotting method works on cv.people.People and plotting styles can be easily updated.")
    pop = sp.Pop(**pars)
    popdict = pop.to_dict()
    cvpopdict = cv.make_synthpop(population=popdict, community_contacts=2)  # array based

    # Actually create the people
    people_pars = dict(
        pop_size=pars.n,
        beta_layer={k: 1.0 for k in 'hswcl'},  # Since this is used to define hat layers exist
        beta=1.0,  # TODO: this is required for plotting (people.plot()), but shouldn't be (in covasim)
    )
    people = cv.People(people_pars, strict=False, uid=cvpopdict['uid'], age=cvpopdict['age'], sex=cvpopdict['sex'])
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.datadir = sp.datadir
    kwargs.figname = f"test_cvpeople_ages_{kwargs.location}_cvpeople"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    # modify some plotting styles
    kwargs.color_1 = '#9966cc'
    kwargs.color_2 = 'indigo'
    kwargs.markersize = 4.5
    fig, ax = sp.plot_ages(people, **kwargs)
    # fig, ax = sp.plot_ages(people)  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check failed.'
    print('Check passed. Figure made.')

    return fig, ax, people


def test_restoring_matplotlib_defaults():
    """
    Test that matplotlib defaults can be restored after plotting_kwargs changes
    them. For example, plotting_kwargs changes the font properties used.
    """
    sp.logger.info("Test that matplotlib defaults can be restored.")
    plkwargs = sp.plotting_kwargs()

    assert mplt.rcParams['font.family'][0] == plkwargs.fontfamily, "Check failed. Instantiating plotting_kwargs did not update the font family for matplotlib.rcParams."
    print("Check passed. matplotlib.rcParams updated font.family to the default fontfamily set in the plotting_kwargs class.")
    assert mplt.rcParams != mplt.rcParamsDefault, "Check failed. matplotlib.rcParams is still the same as matplotlib.rcParamsDefault."
    print("Check passed. matplotlib.rcParams is different from matplotlib.rcParamsDefault.")

    # reset to original matplotlib defaults
    plkwargs.restore_defaults()
    assert mplt.rcParams == mplt.rcParamsDefault, "Check failed. matplotlib.rcParams is not restored to matplotlib.rcParamsDefault."
    print("Check passed. matplotlib.rcParams restored to default matplotlib library values.")


if __name__ == '__main__':

    # run as main and see the examples in action!

    fig0, ax0, pop0 = test_plot_ages(do_show=True)
    fig1, ax1, people1 = test_plot_with_cvpeople(do_show=True, do_save=True)
    test_restoring_matplotlib_defaults()
