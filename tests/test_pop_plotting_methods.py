"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import covasim as cv
import matplotlib as mplt
import cmocean as cmo
import cmasher as cmr
import pytest


# parameters to generate a test population
pars = dict(
    n                               = 2e3,
    rand_seed                       = 123,

    # household_method                = 'fixed_ages',
    smooth_ages                     = True,
    window_length                   = 7,  # window for averaging the age distribution

    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    with_school_types               = 1,

    school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
)
pars = sc.objdict(pars)


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
    # fig, ax = pop.plot_ages(**kwargs)
    fig, ax = pop.plot_ages()

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    popdict = pop.to_dict()
    kwargs.datadir = sp.datadir  # extra information required
    kwargs.figname = f"test_popdict_ages_{kwargs.location}_popdict"
    # fig2, ax2 = sp.plot_ages(popdict, **kwargs)
    fig2, ax2 = sp.plot_ages(popdict)

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
    # fig, ax = sp.plot_ages(people, **kwargs)
    fig, ax = sp.plot_ages(people)

    assert isinstance(fig, mplt.figure.Figure), 'Check failed.'
    print('Check passed. Figure made.')

    return fig, ax, people


@pytest.mark.skip
def test_plot_school_sizes(do_show=False, do_save=False):
    """
    Test that the school size distribution by type plotting method in sp.Pop class works.

    Visually show how the school size distribution generated compares to the data for
    the location being simulated.

    Notes:
        The larger the population size, the better the generated school size
        distributions by school type can match the expected data. If generated
        populations are too small, larger schools will be missed and in
        general there won't be enough schools generated to apply statistical
        tests.

    """
    sp.logger.info("Test that the school size distribution by type plotting method in sp.Pop class works.")
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_school_size_distributions_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    kwargs.rotation = 25
    kwargs.save_dpi = 300
    kwargs.screen_width_factor = 0.30
    kwargs.screen_height_factor = 0.20
    kwargs.hspace = 0.8
    kwargs.bottom = 0.09
    kwargs.location_text_y = 113
    kwargs.keys_to_exclude = ['uv']
    kwargs.cmap = cmr.get_sub_cmap('cmo.curl', 0.08, 1)

    # fig, ax = pop.plot_school_sizes(**kwargs)
    fig, ax = pop.plot_school_sizes()
    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    # works on popdict
    sp.logger.info("Test school size distribution plotting method on popdict.")
    popdict = pop.popdict
    kwargs.datadir = sp.datadir
    kwargs.figname = f"test_school_size_distributions_{kwargs.location}_popdict"
    # fig2, ax2 = sp.plot_school_sizes(popdict, **kwargs)
    fig2, ax2 = sp.plot_school_sizes(popdict)

    assert isinstance(fig2, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    return fig, ax, pop


@pytest.mark.skip
def test_plot_schools_sizes_without_types(do_show=False, do_save=False):
    """Test that without school types, all schools are put together in one group."""
    sp.logger.info("Creating schools where school types are not specified. Test school size distribution plotting method without school types.")
    pars['with_school_types'] = False  # need to rerun the population
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.datadir = sp.datadir
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    kwargs.screen_width_factor = 0.30
    kwargs.screen_height_factor = 0.20
    kwargs.width = 5
    kwargs.height = 3.2
    kwargs.figname = f"test_all_school_size_distributions_{kwargs.location}_pop"
    # fig3, ax3 = pop3.plot_school_sizes(**kwargs)
    fig, ax = pop.plot_school_sizes()

    enrollment_by_school_type = pop.get_enrollment_by_school_type()
    school_types = list(enrollment_by_school_type.keys())

    assert school_types[0] is None and len(school_types) == 1, f"Check 3 failed. School types created: {school_types}."
    # assert isinstance(fig, mplt.figure.Figure), 'Check 3 failed.'
    # print('Check passed. Figure 3 made.')

    return fig, ax, pop


if __name__ == '__main__':

    # run as main and see the examples in action!

    fig0, ax0, pop0 = test_plot_ages(do_show=True)
    fig1, ax1, people1 = test_plot_with_cvpeople(do_show=True, do_save=True)
    fig2, ax2, pop2 = test_plot_school_sizes(do_show=True, do_save=True)
