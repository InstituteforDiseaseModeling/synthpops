"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import covasim as cv
import cmocean as cmo
import pytest


# parameters to generate a test population
pars = dict(
    n                               = 15e3,
    rand_seed                       = 123,
    max_contacts                    = None,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    # location                        = 'Spokane_County',
    # location                        = 'Pierce_County',
    # location                        = 'Yakima_County',
    use_default                     = True,

    household_method                = 'fixed_ages',
    smooth_ages                     = True,
    window_length                   = 7,  # window for averaging the age distribution

    with_industry_code              = 0,
    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    use_two_group_reduction         = 1,
    with_school_types               = 1,

    average_LTCF_degree             = 20,
    ltcf_staff_age_min              = 20,
    ltcf_staff_age_max              = 60,

    school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
    average_class_size              = 20,
    inter_grade_mixing              = 0.1,
    teacher_age_min                 = 25,
    teacher_age_max                 = 75,
    staff_age_min                   = 20,
    staff_age_max                   = 75,

    average_student_teacher_ratio   = 20,
    average_teacher_teacher_degree  = 3,
    average_student_all_staff_ratio = 15,
    average_additional_staff_degree = 20,
)
pars = sc.objdict(pars)


@pytest.mark.parametrize("pars", [pars])
def test_plot_age_distribution_comparison(pars, do_show=False):
    """Test that the age comparison plotting method in sp.Pop class works."""
    sp.logger.info("Test that the age comparison plotting method in sp.Pop class works.")

    pop = sp.Pop(**pars)
    kwargs = sc.dcp(pars)
    kwargs.figname = f"test_pop_ages_{pars['location']}"
    kwargs.do_show = do_show
    fig, ax = pop.plot_age_comparison(**kwargs)
    return fig, ax, pop


@pytest.mark.parametrize("pars", [pars])
def test_plot_with_popdict(pars, do_show=False):
    """
    Test plotting method works on dictionary version of pop object.

    Notes:
        With any popdict, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the age comparison plotting method works on popdict.")
    test_pars = sc.dcp(pars)
    test_pars.location = 'Spokane_County'
    popdict = sp.make_population(**test_pars)
    kwargs = sc.dcp(test_pars)
    kwargs.datadir = sp.datadir
    kwargs.figname = f"test_popdict_ages_{pars['location']}"
    kwargs.do_show = do_show

    fig, ax = sp.plot_age_comparison(popdict, **kwargs)
    return fig, ax, popdict


@pytest.mark.parametrize("pars", [pars])
def test_plot_with_cvpeople(pars, do_show=False, do_save=False):
    """
    Test plotting method works on covasim.people.People object.

    Notes:
        With this pop type, you will need to supply more information to
        tell the method where to look for expected data.
    """
    sp.logger.info("Test that the age comparison plotting method works on cv.people.People")
    popdict = sp.make_population(**pars)  # dict based
    cvpopdict = cv.make_synthpop(population=popdict, community_contacts=10)  # array based

    # Actually create the people
    people_pars = dict(
        pop_size=pars.n,
        beta_layer={k: 1.0 for k in 'hswcl'},  # Since this is used to define hat layers exist
        beta=1.0,  # TODO: this is required for plotting (people.plot()), but shouldn't be (in covasim)
    )
    people = cv.People(people_pars, strict=False, uid=cvpopdict['uid'], age=cvpopdict['age'], sex=cvpopdict['sex'])

    kwargs = sc.objdict(sc.dcp(pars))
    kwargs.datadir = sp.datadir
    kwargs.figname = f"test_cvpeople_ages_{pars['location']}"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    fig, ax = sp.plot_age_comparison(people, **kwargs)
    return fig, ax, people


@pytest.mark.parametrize("pars", [pars])
def test_update_plotting_styles(pars, do_show=False, do_save=False):
    """
    Test plotting method updates with kwargs.
    """
    sp.logger.info("Test that plotting styles get updated.")

    test_pars = sc.dcp(pars)
    test_pars['location'] = 'Spokane_County'
    pop = sp.Pop(**test_pars)
    kwargs = dict(color_1='#9966cc', color_2='indigo', markersize=4.5,
                  # subplot_height=5, subplot_width=8,
                  figname=f"example_ages_{test_pars['location']}",
                  do_save=do_save, do_show=do_show)
    fig, ax = pop.plot_age_comparison(**kwargs)
    return fig, ax, pop


@pytest.mark.parametrize("pars", [pars])
def test_plot_school_sizes_by_type_comparison(pars, do_show=False, do_save=False):
    """Test that the school size distribution by type plotting method in sp.Pop class works."""
    sp.logger.info("Test that the school size distribution by type plotting method in sp.Pop class works.")
    pop = sp.Pop(**pars)
    kwargs = sc.dcp(pars)
    kwargs.figname = f"test_school_size_distributions_{pars['location']}_1"
    # kwargs.figname = f"SchoolSizebyType_{pars['location'].replace('_', ' ').title().replace(' ','')}"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    kwargs.rotation = 20
    kwargs.fontsize = 8.5
    kwargs.save_dpi = 600
    kwargs.screen_width_factor = 0.30
    kwargs.screen_height_factor = 0.20
    kwargs.hspace = 0.8
    kwargs.bottom = 0.09
    kwargs.location_text_y = 113
    kwargs.cmap = cmr.get_sub_map('cmo.curl', 0.08, 1)
    # kwargs.format = 'pdf'
    fig, ax = pop.plot_school_sizes_by_type(**kwargs)

    # # works on popdict
    # popdict = pop.popdict
    # kwargs.datadir = sp.datadir
    # kwargs.figname = f"test_school_size_distributions_{pars['location']}_2"
    # fig2, ax2 = sp.plot_school_sizes_by_type(popdict, **kwargs)

    # # works on popdict
    # pars['with_school_types'] = False
    # pop3 = sp.Pop(**pars)
    # kwargs.datadir = sp.datadir
    # kwargs.figname = f"test_school_size_distributions_{pars['location']}_3"
    # fig3, ax3 = pop3.plot_school_sizes_by_type(**kwargs)

    return fig, ax, pop


if __name__ == '__main__':

    # run as main and see the examples in action!
    # fig0, ax0, pop0 = test_plot_age_distribution_comparison(pars, do_show=True)
    # fig1, ax1, popdict1 = test_plot_with_popdict(pars, do_show=True)
    # fig2, ax2, people2 = test_plot_with_cvpeople(pars, do_show=True, do_save=True)
    fig3, ax3, pop3 = test_update_plotting_styles(pars, do_show=True, do_save=True)
    fig4, ax4, pop4 = test_plot_school_sizes_by_type_comparison(pars, do_show=True, do_save=True)