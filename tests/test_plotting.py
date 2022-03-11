'''
Test plotting functions
'''
import numpy as np
import sciris as sc
import synthpops as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest

mplt_org_backend = mplt.rcParamsDefault['backend']  # interactive backend for user
mplt.use('Agg')


# parameters to generate a test population
pars = sc.objdict(
    n                       = settings.pop_sizes.small_medium,
    rand_seed               = 123,

    smooth_ages             = True,
    household_method        = 'fixed_ages',
    with_facilities         = 1,
    with_non_teaching_staff = 1,
    with_school_types       = 1,

    school_mixing_type      = {'pk': 'age_and_class_clustered',
                               'es': 'age_and_class_clustered',
                               'ms': 'age_and_class_clustered',
                               'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
)


@pytest.fixture(scope="module")
def create_pop():
    return sp.Pop(**pars)


def test_plots(create_pop, do_plot=False):
    ''' Basic plots '''
    if not do_plot:
        plt.switch_backend('agg')  # Plot but don't show
    pop = create_pop  # default parameters, 5k people
    fig1 = pop.plot_people()  # equivalent to pop.to_people().plot()
    fig2 = pop.plot_contacts()  # equivalent to sp.plot_contact_matrix(popdict)
    return [fig1, fig2]


def test_calculate_contact_matrix_errors(create_pop):
    """
    Test that synthpops.plotting.calculate_contact_matrix raises an error when
    density_or_frequency is neither 'density' nor 'frequency'.
    """
    sp.logger.info("Catch ValueError when density_or_frequency is not 'density' or 'frequency'.")
    pop = create_pop
    with pytest.raises(ValueError):
        pop.plot_contacts(density_or_frequency='neither')


def test_catch_pop_type_errors():
    """
    Test that synthpops.plotting methods raise error when pop type is not in
    sp.Pop, dict, or sp.people.People.
    """
    sp.logger.info("Catch NotImplementedError when pop type is invalid.")
    pop = 'not a pop object'

    with pytest.raises(NotImplementedError):
        sp.plot_ages(pop)
    with pytest.raises(NotImplementedError):
        sp.plot_household_sizes(pop)
    with pytest.raises(NotImplementedError):
        sp.plot_ltcf_resident_sizes(pop)
    with pytest.raises(NotImplementedError):
        sp.plot_enrollment_rates_by_age(pop)
    with pytest.raises(NotImplementedError):
        sp.plot_employment_rates_by_age(pop)
    with pytest.raises(NotImplementedError):
        sp.plot_school_sizes(pop)
    with pytest.raises(NotImplementedError):
        sp.plot_workplace_sizes(pop)


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


def test_plot_array():
    """
    Test sp.plot_array() options.
    """
    sp.logger.info("Test sp.plot_array() binned = False option")
    x = np.random.randint(20, size=200)
    kwargs = sc.objdict(binned=False)

    fig, ax = sp.plot_array(x, **kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check sp.plot_array with binned = False failed.'
    print('Check passed. Figure made with sp.plot_array() with unbinned data.')

    hist, bins = np.histogram(x, density=0)
    kwargs.binned = True
    kwargs.generated = hist
    kwargs.value_text = 'hello'
    kwargs.names = {k: k for k in range(20)}
    kwargs.tick_threshold = 5
    fig2, ax2 = sp.plot_array(hist, **kwargs)
    assert isinstance(fig2, mplt.figure.Figure), 'Check sp.plot_array() with other options failed.'
    print('Check passed. Figure made with sp.plot_array() with other options.')


def test_pop_without_plkwargs(create_pop):
    """
    Test that plotting_kwargs will be added to sp.Pop class objects if it does
    not have it already.

    Note:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that plotting_kwargs will be added to sp.Pop object if it does not have it already.")
    pop = create_pop

    # reset plkwargs to None
    pop.plkwargs = None

    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_ages_{kwargs.location}_pop"
    fig, ax = pop.plot_ages(**kwargs)
    # fig, ax = pop.plot_ages()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')


def test_plot_with_sppeople(create_pop, do_show=False, do_save=False):
    """
    Test plotting method works on synthpops.people.People object.

    Notes:
        With this pop type, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the age comparison plotting method works on sp.people.People and plotting styles can be easily updated.")
    pop = create_pop
    people = pop.to_people()
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.datadir = sp.settings.datadir
    kwargs.figname = f"test_ages_{kwargs.location}_sppeople"
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


def summary_plotting_helper(pop, pars, plotting_method_name='plot_ages', do_show=False, do_save=False):
    """
    Generic function to set up test data and logic for different plotting methods.

    Args:
        pars (sc.objdict)          : a dictionary of parameters, modified from default values, to generation the test population
        plotting_method_name (str) : name of the plotting method to test
        do_show (bool)             : If True, show the plot
        do_save (bool)             : If True, save the plot to disk

    Returns:
        A matplotlib figure, matplotlib axes, and a sp.Pop object.

    Note:
        With any popdict, you will need to supply more information to tell the
        method where to look for expected data. Take a look at the second figure
        example to see how to do this.
    """
    sp.logger.info(f"Test that the plotting method: {plotting_method_name} works with sp.Pop object.")
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_{plotting_method_name}_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    if kwargs.do_show:
        plt.switch_backend(mplt_org_backend)

    pop.plotting_method = getattr(pop, plotting_method_name)  # collect the plotting method from pop
    fig, ax = pop.plotting_method(**kwargs)
    # fig, ax = pop.plotting_method()  # to plot without extra information

    assert isinstance(fig, mplt.figure.Figure), 'Check 1 failed.'
    print('Check passed. Figure 1 made.')

    sp.logger.info(f"Test that the plotting method: {plotting_method_name} works with a population dictionary.")
    popdict = pop.to_dict()
    kwargs.datadir = sp.settings.datadir  # extra informaiton required
    kwargs.figname = f"test_{plotting_method_name}_{kwargs.location}_popdict"
    kwargs.do_show = False

    plotting_method = getattr(sp.plotting, plotting_method_name)  # collect the plotting method from sp.plotting
    fig2, ax2 = plotting_method(popdict, **kwargs)
    # fig2, ax2 = plotting_method(popdict)  # to plot without extra information
    if not kwargs.do_show:
        plt.close()
    assert isinstance(fig2, mplt.figure.Figure), 'Check 2 failed.'
    print('Check passed. Figure 2 made.')

    sp.logger.info(f"Test plotting method: {plotting_method_name} without comparison.")
    kwargs.comparison = False
    kwargs.do_show = True
    fig3, ax3 = pop.plotting_method(**kwargs)
    assert isinstance(fig3, mplt.figure.Figure), 'Check 3 failed.'
    print('Check passed. Plotting without comparison.')

    return fig, ax, pop


def test_plot_ages(create_pop, do_show=False, do_save=False):
    """Test that the age comparison plotting method in sp.Pop class works."""
    fig, ax, pop = summary_plotting_helper(create_pop, pars, plotting_method_name='plot_ages', do_show=do_show, do_save=do_save)
    return fig, ax, pop


def test_plot_household_sizes_dist(create_pop, do_show=False, do_save=False):
    """Test that the household sizes comparison plotting method in sp.Pop class works."""
    fig, ax, pop = summary_plotting_helper(create_pop, pars, plotting_method_name='plot_household_sizes', do_show=do_show, do_save=do_save)
    return fig, ax, pop


def test_plot_ltcf_resident_sizes(create_pop, do_show=False, do_save=False):
    """Test that the long term care facility resident sizes comparison plotting method in sp.Pop class works."""
    test_pars = sc.dcp(pars)
    test_pars.n = settings.pop_sizes.large
    fig, ax, pop = summary_plotting_helper(create_pop, test_pars, plotting_method_name='plot_ltcf_resident_sizes', do_show=do_show, do_save=do_save)
    return fig, ax, pop


def test_plot_enrollment_rates_by_age(create_pop, do_show=False, do_save=False):
    """Test that the enrollment rates comparison plotting method in sp.Pop class works."""
    fig, ax, pop = summary_plotting_helper(create_pop, pars, plotting_method_name='plot_enrollment_rates_by_age', do_show=do_show, do_save=do_save)
    return fig, ax, pop


def test_plot_employment_rates_by_age(create_pop, do_show=False, do_save=False):
    """Test that the employment rates comparison plotting method in sp.Pop class works."""
    fig, ax, pop = summary_plotting_helper(create_pop, pars, plotting_method_name='plot_employment_rates_by_age', do_show=do_show, do_save=do_save)
    return fig, ax, pop


def test_plot_workplace_sizes(create_pop, do_show=False, do_save=False):
    """Test that the workplace sizes comparison plotting method in sp.Pop class works."""
    fig, ax, pop = summary_plotting_helper(create_pop, pars, plotting_method_name='plot_workplace_sizes', do_show=do_show, do_save=do_save)
    return fig, ax, pop


def test_household_head_ages_by_size(create_pop, do_show=False, do_save=False):
    """
    Test that the household head age distribution by household size comparison plotting method in sp.Pop class works.

    Args:
        do_show (bool) : If True, show the plot
        do_save (bool) : If True, save the plot to disk

    Returns:
        Matplotlib figure, axes, and pop object.
    """
    sp.logger.info("Test the age distribution of household heads by the household size.")
    pop = create_pop
    kwargs = sc.objdict(sc.mergedicts(pars, pop.loc_pars))
    kwargs.figname = f"test_household_head_ages_by_size_{kwargs.location}_pop"
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    if kwargs.do_show:
        plt.switch_backend(mplt_org_backend)
    fig, ax = pop.plot_household_head_ages_by_size(**kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check failed. Figure not generated.'
    print('Check passed. Figure made.')

    return fig, ax, pop


def test_plot_contact_counts(do_show=False, do_save=False):
    sp.logger.info("Test plot_contact_counts method. Unit test --- for actual use with a sp.Pop object see e2etests/test_workplace_e2e.py.")
    # multiple subplots
    contacts1 = {
        "people_type1": {
            "contact_type1": [2, 3, 4, 3, 4, 3, 4, 2, 1, 2],
            "contact_type2": [10, 22, 11, 12, 9, 9, 8, 10, 11, 10]
        },
        "people_type2": {
            "contact_type1": [1, 3, 4, 3, 5, 3, 4, 4, 1, 2],
            "contact_type2": [10, 21, 11, 11, 10, 9, 8, 10, 11, 10]
        }
    }
    # single subplot
    contacts2 = {
        "people_type": {
            "contact_type": [2, 3, 4, 3, 4, 3, 4, 2, 1, 2]
        }
    }
    params = sc.objdict(do_save=do_save, do_show=do_show, title_prefix='test=true')
    fig1, ax1 = sp.plot_contact_counts(contact_counter=contacts1, **params)
    assert isinstance(fig1, mplt.figure.Figure), 'Check failed. Figure not generated.'
    fig2, ax2 = sp.plot_contact_counts(contact_counter=contacts2, varname="test", **params)
    assert isinstance(fig2, mplt.figure.Figure), 'Check failed. Figure not generated.'
    print('Check passed. Figures made.')


def test_plot_contact_counts_on_pop(create_pop, do_show=False, do_save=False):
    sp.logger.info("Test plot_contact_counts on sp.Pop object.")
    pop = create_pop
    contact_counter = pop.get_contact_counts_by_layer(layer='S')
    assert len(contact_counter['sc_student'])>0, 'contact counter for student should not return 0.'
    assert len(contact_counter['sc_teacher']) > 0, 'contact counter for teacher should not return 0.'
    kwargs = sc.objdict()
    kwargs.do_show = do_show
    kwargs.do_save = do_save

    if kwargs.do_show:
        plt.switch_backend(mplt_org_backend)

    fig, ax = pop.plot_contact_counts(contact_counter, **kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check failed. Figure not generated.'
    return fig, ax, pop


if __name__ == '__main__':

    T = sc.tic()
    pop = sp.Pop(**pars)
    figs = test_plots(create_pop=pop, do_plot=True)
    test_calculate_contact_matrix_errors(create_pop=pop)
    test_catch_pop_type_errors()
    test_restoring_matplotlib_defaults()
    test_plot_array()
    test_pop_without_plkwargs(create_pop=pop)
    fig0, ax0, pop0 = test_plot_ages(create_pop=pop, do_show=True)
    fig1, ax1, people1 = test_plot_with_sppeople(create_pop=pop, do_show=True, do_save=True)
    fig2, ax2, pop2 = test_plot_household_sizes_dist(create_pop=pop, do_show=True)
    fig3, ax3, pop3 = test_plot_ltcf_resident_sizes(create_pop=pop, do_show=True)
    fig4, ax4, pop4 = test_plot_enrollment_rates_by_age(create_pop=pop, do_show=True)
    fig5, ax5, pop5 = test_plot_employment_rates_by_age(create_pop=pop, do_show=True)
    fig6, ax6, pop6 = test_plot_workplace_sizes(create_pop=pop, do_show=True)
    fig7, ax7, pop7 = test_household_head_ages_by_size(create_pop=pop, do_show=True)
    test_plot_contact_counts(do_show=True)
    test_plot_contact_counts_on_pop(create_pop=pop, do_show=True)

    sc.toc(T)
    print('Done.')
